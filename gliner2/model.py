import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from gliner.modeling.span_rep import SpanRepLayer
from gliner2.layers import CountLSTMoE, CountLSTM, create_mlp, CountLSTMv2
from gliner2.processor import SchemaTransformer
from safetensors.torch import save_file, load_file
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoModel,
    AutoConfig,
    AutoTokenizer,
)


# ---------------------------
# ExtractorConfig Definition
# ---------------------------
class ExtractorConfig(PretrainedConfig):
    model_type = "extractor"

    def __init__(self, model_name="bert-base-uncased", max_width=8, counting_layer="count_lstm", **kwargs):
        """
        Configuration for the Extractor model.

        Args:
            model_name (str): Pretrained model name for the encoder.
            max_width (int): Maximum width for span representations.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_width = max_width
        self.counting_layer = counting_layer


# ---------------------------
# Extractor Model
# ---------------------------
class Extractor(PreTrainedModel):
    config_class = ExtractorConfig

    def __init__(self, config, encoder_config=None, tokenizer=None):
        """
        Initializes the Extractor model.

        Args:
            config (ExtractorConfig): Model configuration.
            encoder_config (Optional[dict or PretrainedConfig]): Configuration for the encoder.
            tokenizer (Optional[PreTrainedTokenizer]): Tokenizer for the processor.
        """
        super().__init__(config)
        self.config = config
        self.max_width = config.max_width

        # Initialize the SchemaTransformer. If a tokenizer is provided, use it.
        if tokenizer is not None:
            self.processor = SchemaTransformer(tokenizer=tokenizer)
        else:
            self.processor = SchemaTransformer(config.model_name)

        # Load the encoder model.
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
        # Resize token embeddings using the processor's tokenizer.
        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size

        # Span representation layer.
        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=self.hidden_size,
            max_width=self.max_width,
            dropout=0.1,
        )
        # Classifier for classification tasks.
        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )
        # Layers for hierarchical schema count prediction.
        self.count_pred = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=20,
            dropout=0.,
            activation="relu",
            add_layer_norm=False
        )
        # Count embedding module.
        if config.counting_layer == "count_lstm":
            print("Using standard CountLSTM")
            self.count_embed = CountLSTM(self.hidden_size)
        elif config.counting_layer == "count_lstm_moe":
            print("Using Mixture of Experts for CountLSTM")
            self.count_embed = CountLSTMoE(
                hidden_size=self.hidden_size,
                n_experts=4,
                ffn_mult=2,
                dropout=0.1
            )
        elif config.counting_layer == "count_lstm_v2":
            print("Using CountLSTMv2")
            self.count_embed = CountLSTMv2(hidden_size=self.hidden_size)

    # ---------------------------
    # Original Methods
    # ---------------------------
    def compute_span_rep(self, token_embeddings: torch.Tensor) -> dict:
        text_length = len(token_embeddings)
        device = next(self.encoder.parameters()).device

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((-1, -1))  # Explicitly mark invalid spans

        spans_idx = torch.LongTensor([spans_idx]).to(device)  # shape: [1, num_spans, 2]

        # Mask invalid spans (where either start or end == -1)
        span_mask = (spans_idx[:, :, 0] == -1) | (spans_idx[:, :, 1] == -1)
        span_mask = span_mask.to(device)

        # Replace invalid spans with dummy (0, 0) for safe indexing
        safe_spans_idx = torch.where(
            span_mask.unsqueeze(-1),
            torch.zeros_like(spans_idx),
            spans_idx
        )

        # Compute span representations (batch size = 1)
        span_rep = self.span_rep(token_embeddings.unsqueeze(0), safe_spans_idx).squeeze(0)  # [num_spans, hidden_dim]

        return {
            "span_rep": span_rep,
            "spans_idx": spans_idx,
            "span_mask": span_mask,
        }

    def classification_loss(self, embs_per_schema: list, task_types: list, structure_labels: list) -> torch.Tensor:
        """
        Computes binary classification loss for classification tasks.
        """
        cls_embeds = []
        binary_labels = []
        for i, task_type in enumerate(task_types):
            if task_type == "classifications":
                schema_embs = torch.stack(embs_per_schema[i], dim=0)  # [num_tokens, hidden_size]
                # Exclude the first token ([P]) from classification.
                cls_embeds.append(schema_embs[1:])
                binary_labels.extend(structure_labels[i])

        if not cls_embeds:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        cls_embeds = torch.cat(cls_embeds, dim=0)  # [total_num_tokens, hidden_size]
        logits = self.classifier(cls_embeds)  # [total_num_tokens, 1]
        binary_labels = torch.FloatTensor(binary_labels).to(logits.device)

        # --- DYNAMIC POSITIVE WEIGHTING ---
        # Calculate the ratio of negative/positive examples for current batch
        # num_positives = (binary_labels == 1).sum()
        # num_negatives = (binary_labels == 0).sum()
        # To avoid division by zero
        # pos_weight = (num_negatives / (num_positives + 1e-6)).clamp(min=1.0, max=5.0)

        # Create BCE loss with batch-specific pos_weight
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")  # pos_weight=pos_weight)

        return loss_fn(logits.squeeze(-1), binary_labels)

    def compute_struct_loss(self, h, pc_embeddings, structures, span_mask,
                            masking_rate=0.5):
        """
        Enhanced structure loss with random negative span masking to improve recall
        by addressing the unlabeled entity problem.

        Args:
            h: Span representations
            pc_embeddings: Schema embeddings
            structures: Ground truth structure labels
            span_mask: Mask for valid spans
            masking_rate: Probability of masking negative spans (0.0 to 1.0)
        """
        gold_count_val = min(structures[0], 19)
        struct_proj = self.count_embed(pc_embeddings[1:], gold_count_val)
        scores = torch.einsum('lkd,bpd->bplk', h, struct_proj)

        # Create label tensor matching the shape of scores
        labs = torch.zeros_like(scores)
        for i in range(gold_count_val):
            gold_spans = structures[1][i]
            for k, span in enumerate(gold_spans):
                if span is None or span == (-1, -1):
                    continue
                if isinstance(span, tuple):
                    start, end = span
                    width = end - start
                    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                        labs[i, k, start, width] = 1
                elif isinstance(span, list):
                    for sub_span in span:
                        if sub_span is None or sub_span == (-1, -1):
                            continue
                        start, end = sub_span
                        width = end - start
                        if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                            labs[i, k, start, width] = 1

        # Apply random masking to negative spans only
        if masking_rate > 0.0:
            # Identify negative spans (where labels are 0)
            negative_spans = (labs == 0)

            # Generate random mask for negative spans
            random_mask = torch.rand_like(scores) < masking_rate

            # Apply masking only to negative spans
            spans_to_mask = negative_spans & random_mask

            # Create loss computation mask (1 = compute loss, 0 = ignore)
            loss_computation_mask = (~spans_to_mask).float()
        else:
            loss_computation_mask = torch.ones_like(scores)

        # Compute loss with masking applied
        loss_struct = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        loss_struct = loss_struct * loss_computation_mask
        loss_struct = loss_struct.view(loss_struct.shape[0], loss_struct.shape[1], -1) * (~span_mask[0]).float()
        loss_struct = loss_struct.sum()

        return loss_struct

    def process_record(self, record: dict) -> dict:
        """
        Processes a single record to obtain inputs, compute embeddings, span representations, and losses.
        """
        outputs = self.processor.process_record(self.encoder, record)

        tokens = outputs["text_tokens"]
        embs_per_schema = outputs["embs_per_schema"]
        structure_labels = outputs["structure_labels"]
        task_types = outputs["task_types"]
        token_embeddings = outputs["token_embeddings"]

        device = next(self.parameters()).device

        span_rep = self.compute_span_rep(token_embeddings)
        classification_loss_val = self.classification_loss(embs_per_schema, task_types, structure_labels)

        structure_loss = 0.0
        count_loss = 0.0
        all_counts = []
        all_p_embs = []
        for schema_idx in range(len(embs_per_schema)):
            if task_types[schema_idx] != "classifications":
                schema_emb = torch.stack(embs_per_schema[schema_idx])
                structure = structure_labels[schema_idx]
                # if count is 0, loss is 0
                if structure[0] == 0:
                    loss_struct = torch.tensor(0.0, device=device)
                else:
                    loss_struct = self.compute_struct_loss(
                        span_rep["span_rep"], schema_emb, structure, span_rep["span_mask"]
                    )
                structure_loss += loss_struct

                if task_types[schema_idx] != "entities":
                    # For hierarchical schemas, we need to predict counts.
                    all_counts.append(min(structure[0], 19))
                    all_p_embs.append(schema_emb[0])

        if all_counts and all_p_embs:
            all_counts = torch.LongTensor(all_counts).to(device)
            all_p_embs = torch.stack(all_p_embs, dim=0)
            count_loss = F.cross_entropy(self.count_pred(all_p_embs), all_counts, reduction="sum")

        total_loss = classification_loss_val + structure_loss + count_loss

        return {
            "text_tokens": tokens,
            "embs_per_schema": embs_per_schema,
            "structure_labels": structure_labels,
            "task_types": task_types,
            "token_embeddings": token_embeddings,
            "span_rep": span_rep,
            "loss": total_loss,
            "loss_fine_grained": {
                "classification_loss": classification_loss_val,
                "structure_loss": structure_loss,
                "count_loss": count_loss,
            }
        }

    # ---------------------------
    # Hugging Face Compatibility Methods
    # ---------------------------
    def push_to_hub(self, repo_id: str, private: bool = True):
        """
        Push the model (configuration, weights, and tokenizer) to the Hugging Face Hub.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            super().push_to_hub(repo_id=repo_id, save_dir=tmp_dir, private=private)
            # Note: the tokenizer used is from the processor.
            self.processor.tokenizer.push_to_hub(repo_id)

    @classmethod
    def from_pretrained(cls, repo_or_dir, **kwargs):
        """
        Load the model from a directory or repository on the Hugging Face Hub.
        """
        from huggingface_hub import hf_hub_download  # ensure this is available

        def download_or_local(repo_or_dir, filename):
            if os.path.isdir(repo_or_dir):
                return os.path.join(repo_or_dir, filename)
            else:
                return hf_hub_download(repo_or_dir, filename)

        # Load the model configuration.
        config_path = download_or_local(repo_or_dir, "config.json")
        config = cls.config_class.from_pretrained(config_path)

        # Load the encoder configuration.
        encoder_config_path = download_or_local(repo_or_dir, "encoder_config/config.json")
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)

        # Load the tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)

        # Initialize the model.
        model = cls(config, encoder_config=encoder_config, tokenizer=tokenizer)

        # Load model weights.
        try:
            model_path = download_or_local(repo_or_dir, "model.safetensors")
            model_state_dict = load_file(model_path)
        except Exception:
            model_path = download_or_local(repo_or_dir, "pytorch_model.bin")
            model_state_dict = torch.load(model_path, map_location="cpu")

        # if shape is not equal, fill additional tokens embedding to the state dict given the model
        try:
            if model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[0] != \
                    model.encoder.embeddings.word_embeddings.weight.shape[0]:
                model_state_dict["encoder.embeddings.word_embeddings.weight"] = torch.cat(
                    [
                        model_state_dict["encoder.embeddings.word_embeddings.weight"],
                        torch.randn(model.encoder.embeddings.word_embeddings.weight.shape[0] -
                                    model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[0],
                                    model_state_dict["encoder.embeddings.word_embeddings.weight"].shape[1]) * 0.02
                    ],
                    dim=0
                )
        except KeyError:
            pass

        model.load_state_dict(model_state_dict)

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model configuration, encoder configuration, state dict (using safetensors),
        and tokenizer to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # Save the main config.
        self.config.save_pretrained(save_directory)

        # Save encoder configuration.
        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        # Save model weights using safetensors.
        model_save_path = os.path.join(save_directory, "model.safetensors")
        save_file(self.state_dict(), model_save_path)

        # Save the tokenizer from the processor.
        self.processor.tokenizer.save_pretrained(save_directory)
