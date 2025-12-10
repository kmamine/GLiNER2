"""
GLiNER2 World-Class Trainer
===========================

Production-grade training infrastructure with flexible data input.

Supported Data Formats:
-----------------------
1. Single JSONL file path (str or Path)
2. List of JSONL file paths
3. List of InputExample objects
4. TrainingDataset object
5. List of raw dict records ({"input": ..., "output": ...} format)

Basic Examples:
--------------
    >>> from gliner2.training.data import InputExample, TrainingDataset
    >>> from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer
    >>>
>>> # 1. From list of InputExample
    >>> examples = [
    ...     InputExample(text="John works at Google.", entities={"person": ["John"], "company": ["Google"]}),
    ...     InputExample(text="Apple released iPhone.", entities={"company": ["Apple"], "product": ["iPhone"]}),
    ... ]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)
    >>>
>>> # 2. From JSONL file(s)
    >>> trainer.train(train_data="train.jsonl")
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])
>>>
>>> # 3. From TrainingDataset
>>> dataset = TrainingDataset.load("train.jsonl")
>>> trainer.train(train_data=dataset)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

from gliner2.processor import SchemaTransformer, SamplingConfig

# Import training data classes
from gliner2.training.data import (
    InputExample, TrainingDataset, ValidationError,
    DataFormat, detect_data_format, DataLoader_Factory, TrainDataInput
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Parameters
    ----------
    output_dir : str
        Directory for saving checkpoints and logs.
    experiment_name : str
        Name of the experiment (used for logging).
    num_epochs : int
        Number of training epochs.
    max_steps : int
        Maximum training steps (-1 = determined by epochs).
    batch_size : int
        Training batch size per device.
    eval_batch_size : int
        Evaluation batch size.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    encoder_lr : float
        Learning rate for encoder parameters.
    task_lr : float
        Learning rate for task-specific parameters.
    weight_decay : float
        Weight decay for AdamW optimizer.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    scheduler_type : str
        LR scheduler type: "linear", "cosine", "cosine_restarts", "constant".
    warmup_ratio : float
        Warmup ratio (portion of total steps).
    warmup_steps : int
        Explicit warmup steps (overrides warmup_ratio if > 0).
    fp16 : bool
        Use FP16 mixed precision.
    bf16 : bool
        Use BF16 mixed precision.
    save_strategy : str
        When to save: "epoch", "steps", or "no".
    save_steps : int
        Save every N steps (if save_strategy="steps").
    save_total_limit : int
        Maximum checkpoints to keep.
    save_best : bool
        Save best model based on metric.
    metric_for_best : str
        Metric to use for best model selection.
    greater_is_better : bool
        Whether higher metric is better.
    eval_strategy : str
        When to evaluate: "epoch", "steps", or "no".
    eval_steps : int
        Evaluate every N steps (if eval_strategy="steps").
    logging_steps : int
        Log every N steps.
    report_to : List[str]
        Logging backends: "tensorboard", "wandb".
    early_stopping : bool
        Enable early stopping.
    early_stopping_patience : int
        Patience for early stopping.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed.
    validate_data : bool
        Validate training data before training.
    strict_validation : bool
        Check entity spans exist in text.
    """
    output_dir: str = "./output"
    experiment_name: str = "gliner2"
    num_epochs: int = 10
    max_steps: int = -1
    batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    encoder_lr: float = 1e-5
    task_lr: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    num_cycles: float = 0.5
    fp16: bool = True
    bf16: bool = False
    save_strategy: str = "epoch"
    save_steps: int = 500
    save_total_limit: int = 3
    save_best: bool = True
    metric_for_best: str = "eval_loss"
    greater_is_better: bool = False
    eval_strategy: str = "epoch"
    eval_steps: int = 500
    logging_steps: int = 50
    logging_first_step: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    seed: int = 42
    deterministic: bool = False
    gradient_checkpointing: bool = False
    local_rank: int = -1
    debug: bool = False
    max_train_samples: int = -1
    max_eval_samples: int = -1
    validate_data: bool = True
    strict_validation: bool = False

    def __post_init__(self):
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if self.bf16 and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported, falling back to fp16")
            self.bf16 = False
            self.fp16 = True

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# Dataset
# =============================================================================

class ExtractorDataset(Dataset):
    """
    Dataset for GLiNER2 training with multi-format support.

    Supports all formats through DataLoader_Factory:
    - JSONL file path(s)
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records
    
    Examples
    --------
    >>> # From JSONL
    >>> dataset = ExtractorDataset("train.jsonl")
    
    >>> # From multiple JSONL files
    >>> dataset = ExtractorDataset(["train1.jsonl", "train2.jsonl"])
    
    >>> # From InputExample list
    >>> dataset = ExtractorDataset(examples)
    """

    def __init__(
            self,
            data: TrainDataInput,
            max_samples: int = -1,
            shuffle: bool = True,
            seed: int = 42,
            validate: bool = False,
            strict_validation: bool = False,
    ):
        """
        Initialize dataset from various input formats.

        Parameters
        ----------
        data : TrainDataInput
            Training data in any supported format.
        max_samples : int, default=-1
            Maximum samples to use (-1 = all).
        shuffle : bool, default=True
            Whether to shuffle the data.
        seed : int, default=42
            Random seed for shuffling.
        validate : bool, default=False
            Whether to validate the data.
        strict_validation : bool, default=False
            If True, checks that entity spans exist in text.
        """
        self.data = DataLoader_Factory.load(
            data=data,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            validate=validate,
            strict_validation=strict_validation,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        record = self.data[idx]
        # Handle both formats
        if "input" in record:
            return record["input"], record["output"]
        else:
            return record["text"], record["schema"]

    # Factory methods for explicit creation
    @classmethod
    def from_jsonl(cls, paths: Union[str, Path, List], **kwargs) -> 'ExtractorDataset':
        """Create from JSONL file(s)."""
        return cls(paths, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[InputExample], **kwargs) -> 'ExtractorDataset':
        """Create from list of InputExample."""
        return cls(examples, **kwargs)

    @classmethod
    def from_training_dataset(cls, dataset: TrainingDataset, **kwargs) -> 'ExtractorDataset':
        """Create from TrainingDataset."""
        return cls(dataset, **kwargs)

    @classmethod
    def from_dicts(cls, dicts: List[Dict], **kwargs) -> 'ExtractorDataset':
        """Create from list of dicts."""
        return cls(dicts, **kwargs)


# =============================================================================
# Collator
# =============================================================================

class ExtractorCollator:
    """Data collator that converts raw records to model inputs."""

    def __init__(self, processor: SchemaTransformer, is_training: bool = True):
        self.processor = processor
        self.is_training = is_training

    def __call__(self, batch: List[Tuple[str, Dict]]):
        """
        Convert batch of (text, schema) tuples to PreprocessedBatch.
        
        Args:
            batch: List of (text, schema) tuples from dataset
            
        Returns:
            PreprocessedBatch ready for model.forward()
        """
        if self.is_training:
            return self.processor.collate_fn_train(batch)
        else:
            return self.processor.collate_fn_inference(batch)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    classification_loss: float = 0.0
    structure_loss: float = 0.0
    count_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    step: int = 0
    samples_seen: int = 0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Scheduler Factory
# =============================================================================

def get_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps, num_cycles=0.5):
    """Create learning rate scheduler."""
    def lr_lambda_linear(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    def lr_lambda_cosine(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda_cosine_restarts(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))))

    def lr_lambda_constant(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0

    schedulers = {
        "linear": lr_lambda_linear,
        "cosine": lr_lambda_cosine,
        "cosine_restarts": lr_lambda_cosine_restarts,
        "constant": lr_lambda_constant,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return LambdaLR(optimizer, schedulers[scheduler_type])


# =============================================================================
# Main Trainer
# =============================================================================

class GLiNER2Trainer:
    """
    World-class trainer for GLiNER2 with flexible multi-format data input.

    Parameters
    ----------
    model : nn.Module
        The GLiNER2 model to train.
    config : TrainingConfig
        Training configuration.
    processor : SchemaTransformer, optional
        Schema processor. If None, uses model.processor.
    train_data : TrainDataInput, optional
        Training data (can be provided here or in train()).
    eval_data : TrainDataInput, optional
        Evaluation data.
    compute_metrics : Callable, optional
        Custom metrics function.

    Supported Data Formats
    ----------------------
    - Single JSONL file path (str or Path)
    - List of JSONL file paths
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records

    Examples
    --------
    >>> # With InputExample list
    >>> examples = [InputExample(...), InputExample(...)]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)

    >>> # With JSONL file
    >>> trainer.train(train_data="train.jsonl")

    >>> # With multiple JSONL files
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])

    >>> # With TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> trainer.train(train_data=dataset)
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            processor: SchemaTransformer = None,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
            compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.processor = processor or getattr(model, 'processor', None)
        if self.processor is None:
            raise ValueError("Processor must be provided or model must have .processor attribute")

        self.train_data = train_data
        self.eval_data = eval_data
        self.compute_metrics = compute_metrics

        self._setup_seed()
        self._setup_device()
        self._setup_output_dir()
        self._setup_logging()

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.patience_counter = 0
        self.train_metrics_history = []
        self.eval_metrics_history = []

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.wandb_run = None

    def _setup_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def _setup_device(self):
        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device("cuda", self.config.local_rank)
            self.is_distributed = True
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_distributed = False
        else:
            self.device = torch.device("cpu")
            self.is_distributed = False
            if self.config.fp16 or self.config.bf16:
                logger.warning("Mixed precision disabled on CPU")
                self.config.fp16 = False
                self.config.bf16 = False
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def _setup_output_dir(self):
        self.output_dir = Path(self.config.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.config.save(str(self.output_dir / "training_config.json"))

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.is_main_process else logging.WARNING,
        )
        self.tb_writer = None
        if "tensorboard" in self.config.report_to and self.is_main_process:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.logs_dir))
            except ImportError:
                logger.warning("TensorBoard not available")

        if "wandb" in self.config.report_to and self.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project or self.config.experiment_name,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name or f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=asdict(self.config),
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes,
                    dir=str(self.output_dir),
                    resume="allow",
                )
                wandb.watch(self.model, log="gradients", log_freq=self.config.logging_steps)
                logger.info(f"Weights & Biases initialized: {self.wandb_run.url}")
            except ImportError:
                logger.warning("wandb not available")

    @property
    def is_main_process(self) -> bool:
        return self.config.local_rank <= 0

    def _prepare_data(self, data: TrainDataInput, is_train: bool = True) -> ExtractorDataset:
        """Convert any supported data format to ExtractorDataset."""
        if data is None:
            return None

        if isinstance(data, ExtractorDataset):
            return data

        max_samples = self.config.max_train_samples if is_train else self.config.max_eval_samples

        return ExtractorDataset(
            data=data,
            max_samples=max_samples,
            shuffle=is_train,
            seed=self.config.seed,
            validate=self.config.validate_data if is_train else False,
            strict_validation=self.config.strict_validation,
        )

    def _create_optimizer(self) -> AdamW:
        encoder_params = []
        task_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                task_params.append(param)

        return AdamW(
            [
                {"params": encoder_params, "lr": self.config.encoder_lr, "weight_decay": self.config.weight_decay},
                {"params": task_params, "lr": self.config.task_lr, "weight_decay": self.config.weight_decay},
            ],
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_dataloader(self, dataset: ExtractorDataset, batch_size: int, shuffle: bool = True, is_training: bool = True) -> DataLoader:
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        collator = ExtractorCollator(self.processor, is_training=is_training)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            collate_fn=collator,
            drop_last=is_training,
            persistent_workers=self.config.num_workers > 0,
        )

    def train(
            self,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Parameters
        ----------
        train_data : TrainDataInput, optional
            Training data. Supports all formats:
            - str/Path: JSONL file path
            - List[str/Path]: Multiple JSONL files
            - List[InputExample]: List of examples
            - TrainingDataset: Dataset object
            - List[Dict]: Raw records

        eval_data : TrainDataInput, optional
            Evaluation data (same formats supported).

        Returns
        -------
        Dict[str, Any]
            Training summary with metrics history.
        """
        # Prepare datasets
        train_data = train_data or self.train_data
        eval_data = eval_data or self.eval_data

        if train_data is None:
            raise ValueError("No training data provided")

        train_dataset = self._prepare_data(train_data, is_train=True)
        eval_dataset = self._prepare_data(eval_data, is_train=False) if eval_data else None

        train_loader = self._create_dataloader(train_dataset, self.config.batch_size, shuffle=True, is_training=True)

        # Calculate steps
        num_update_steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
        else:
            max_steps = num_update_steps_per_epoch * self.config.num_epochs
            num_epochs = self.config.num_epochs

        warmup_steps = self.config.warmup_steps or int(max_steps * self.config.warmup_ratio)

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_type, max_steps, warmup_steps, self.config.num_cycles)

        # Mixed precision
        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self.scaler = GradScaler(enabled=self.config.fp16)

        if self.config.gradient_checkpointing:
            self.model.encoder.gradient_checkpointing_enable()

        # Logging
        logger.info("***** Running Training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.config.effective_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Warmup steps = {warmup_steps}")

        # Training state
        self.model.train()
        self.processor.change_mode(is_training=True)
        self.global_step = 0
        self.epoch = 0
        tr_loss = 0.0

        start_time = time.time()
        samples_seen = 0

        progress_bar = tqdm(total=max_steps, desc="Training", disable=not self.is_main_process)

        for epoch in range(num_epochs):
            self.epoch = epoch

            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                samples_seen += len(batch)

                with autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = self.model(batch)
                    loss = outputs["total_loss"]

                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()
                epoch_steps += 1

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    if self.config.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - start_time
                        metrics = TrainingMetrics(
                            loss=tr_loss / self.config.logging_steps,
                            classification_loss=outputs.get("classification_loss", torch.tensor(0)).item(),
                            structure_loss=outputs.get("structure_loss", torch.tensor(0)).item(),
                            count_loss=outputs.get("count_loss", torch.tensor(0)).item(),
                            learning_rate=self.scheduler.get_last_lr()[0],
                            epoch=epoch + step / len(train_loader),
                            step=self.global_step,
                            samples_seen=samples_seen,
                            throughput=samples_seen / elapsed,
                        )
                        self._log_metrics(metrics, prefix="train")
                        tr_loss = 0.0

                    if self.config.eval_strategy == "steps" and self.global_step % self.config.eval_steps == 0 and eval_dataset:
                        self._evaluate(eval_dataset)
                        self.model.train()

                    if self.config.save_strategy == "steps" and self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{self.global_step}")

                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        "grad": f"{grad_norm:.2f}",
                    })

                    if self.global_step >= max_steps:
                        break

            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / epoch_steps:.4f}")

            if self.config.eval_strategy == "epoch" and eval_dataset:
                eval_metrics = self._evaluate(eval_dataset)
                self.model.train()
                if self.config.early_stopping and self._check_early_stopping(eval_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            if self.config.save_strategy == "epoch":
                self._save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

            if self.global_step >= max_steps:
                break

        progress_bar.close()

        if self.is_main_process:
            self._save_checkpoint("final")
            if self.tb_writer:
                self.tb_writer.close()
            if self.wandb_run:
                import wandb
                wandb.summary["best_metric"] = self.best_metric
                wandb.summary["total_steps"] = self.global_step
                wandb.finish()

        total_time = time.time() - start_time
        return {
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "total_time_seconds": total_time,
            "samples_per_second": samples_seen / total_time,
            "best_metric": self.best_metric,
            "train_metrics_history": self.train_metrics_history,
            "eval_metrics_history": self.eval_metrics_history,
        }

    def _evaluate(self, eval_dataset: ExtractorDataset) -> Dict[str, float]:
        logger.info("Running evaluation...")
        self.model.eval()
        self.processor.change_mode(is_training=False)

        eval_loader = self._create_dataloader(eval_dataset, self.config.eval_batch_size, shuffle=False, is_training=False)

        total_loss = 0.0
        total_cls_loss = 0.0
        total_struct_loss = 0.0
        total_count_loss = 0.0
        num_batches = 0

        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.is_main_process):
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = self.model(batch)

                total_loss += outputs["total_loss"].item()
                total_cls_loss += outputs.get("classification_loss", torch.tensor(0)).item()
                total_struct_loss += outputs.get("structure_loss", torch.tensor(0)).item()
                total_count_loss += outputs.get("count_loss", torch.tensor(0)).item()
                num_batches += 1

        metrics = {
            "eval_loss": total_loss / num_batches,
            "eval_classification_loss": total_cls_loss / num_batches,
            "eval_structure_loss": total_struct_loss / num_batches,
            "eval_count_loss": total_count_loss / num_batches,
            "step": self.global_step,
            "epoch": self.epoch,
        }

        if self.compute_metrics:
            metrics.update(self.compute_metrics(self.model, eval_dataset))

        self._log_metrics(metrics, prefix="eval")
        self.eval_metrics_history.append(metrics)

        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        is_best = (
            (self.config.greater_is_better and metric_value > self.best_metric) or
            (not self.config.greater_is_better and metric_value < self.best_metric)
        )

        if is_best:
            self.best_metric = metric_value
            if self.config.save_best:
                self._save_checkpoint("best")
            logger.info(f"New best {self.config.metric_for_best}: {self.best_metric:.4f}")

        return metrics

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        if self.config.greater_is_better:
            improved = metric_value > self.best_metric + self.config.early_stopping_threshold
        else:
            improved = metric_value < self.best_metric - self.config.early_stopping_threshold

        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _log_metrics(self, metrics: Union[Dict, TrainingMetrics], prefix: str = ""):
        if isinstance(metrics, TrainingMetrics):
            metrics = metrics.to_dict()

        if self.is_main_process:
            log_str = f"[Step {self.global_step}]"
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_str += f" {prefix}_{key}: {value:.4f}"
            logger.info(log_str)

        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}/{key}", value, self.global_step)

        if self.wandb_run:
            import wandb
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            if wandb_metrics:
                wandb.log(wandb_metrics, step=self.global_step)

        if prefix == "train":
            self.train_metrics_history.append(metrics)

    def _save_checkpoint(self, name: str):
        if not self.is_main_process:
            return

        checkpoint_dir = self.checkpoints_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        self.model.save_pretrained(str(checkpoint_dir))

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if self.scaler:
            state["scaler_state"] = self.scaler.state_dict()

        torch.save(state, checkpoint_dir / "training_state.pt")
        logger.info(f"Saved checkpoint: {name}")

        if self.wandb_run and name in ["best", "final"]:
            try:
                import wandb
                artifact = wandb.Artifact(f"model-{self.config.experiment_name}-{name}", type="model")
                artifact.add_dir(str(checkpoint_dir))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to save to wandb: {e}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: x.stat().st_mtime,
        )
        protected = {"best", "final"}
        checkpoints = [c for c in checkpoints if c.name not in protected]

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest.name}")

    def resume_from_checkpoint(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)
        self.model = self.model.__class__.from_pretrained(str(checkpoint_dir))
        self.model.to(self.device)

        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state["best_metric"]
            if self.optimizer and "optimizer_state" in state:
                self.optimizer.load_state_dict(state["optimizer_state"])
            if self.scheduler and "scheduler_state" in state:
                self.scheduler.load_state_dict(state["scheduler_state"])
            if self.scaler and "scaler_state" in state:
                self.scaler.load_state_dict(state["scaler_state"])

        logger.info(f"Resumed from checkpoint: {checkpoint_path} (step {self.global_step})")


# =============================================================================
# Convenience Functions
# =============================================================================

def train_gliner2(
        model_path: str,
        train_data: TrainDataInput,
        output_dir: str = "./output",
        eval_data: TrainDataInput = None,
        **config_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for training GLiNER2.

    Parameters
    ----------
    model_path : str
        Path to pretrained model.
    train_data : TrainDataInput
        Training data in any supported format:
        - JSONL path(s)
        - List of InputExample
        - TrainingDataset
        - List of dicts
    output_dir : str
        Output directory for checkpoints.
    eval_data : TrainDataInput, optional
        Evaluation data.
    **config_kwargs
        Additional TrainingConfig parameters.

    Returns
    -------
    Dict[str, Any]
        Training results.

    Examples
    --------
    >>> # Train with JSONL file
    >>> results = train_gliner2("model-path", "train.jsonl", num_epochs=10)

    >>> # Train with multiple JSONL files
    >>> results = train_gliner2("model-path", ["train1.jsonl", "train2.jsonl"])

    >>> # Train with InputExample list
    >>> examples = [InputExample(...), ...]
    >>> results = train_gliner2("model-path", examples)

    >>> # Train with TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> results = train_gliner2("model-path", dataset)
    """
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_path)
    config = TrainingConfig(output_dir=output_dir, **config_kwargs)

    trainer = GLiNER2Trainer(model=model, config=config)
    return trainer.train(train_data=train_data, eval_data=eval_data)