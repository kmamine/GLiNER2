"""
Custom LoRA (Low-Rank Adaptation) Implementation for GLiNER2
=============================================================

Parameter-efficient fine-tuning by injecting trainable low-rank matrices
into frozen linear layers of the encoder.

Based on: "LoRA: Low-Rank Adaptation of Large Language Models"
Paper: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA Configuration
# =============================================================================

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA parameter-efficient fine-tuning.
    
    Parameters
    ----------
    enabled : bool
        Whether LoRA is enabled.
    r : int
        Rank of the low-rank decomposition (bottleneck dimension).
        Higher r = more parameters but better approximation.
        Typical values: 4, 8, 16, 32, 64.
    alpha : float
        Scaling factor for LoRA updates. Final scaling is alpha/r.
        Typical values: 8, 16, 32 (often 2*r).
    dropout : float
        Dropout probability applied to LoRA path.
    target_modules : List[str]
        Names of modules to apply LoRA to. Uses substring matching.
        Examples:
        - ["query", "key", "value"] - matches query_proj, key_proj, value_proj
        - ["query_proj", "key_proj", "value_proj"] - exact match also works
        - ["dense"] - matches all layers named 'dense'
        Applied to encoder only.
    """
    enabled: bool = False
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["query", "key", "value"])
    
    def __post_init__(self):
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {self.r}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
        if self.enabled and not self.target_modules:
            raise ValueError("target_modules cannot be empty when LoRA is enabled")


# =============================================================================
# LoRA Layer
# =============================================================================

class LoRALayer(nn.Module):
    """
    LoRA-enhanced Linear layer.
    
    Computes: output = W*x + (B*A*x) * scaling
    Where:
        - W is the frozen original weight
        - A, B are trainable low-rank matrices
        - scaling = alpha / r
    
    Parameters
    ----------
    base_layer : nn.Linear
        Original linear layer (will be frozen).
    r : int
        Rank of low-rank decomposition.
    alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Store frozen base layer
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA low-rank matrices
        # A: (r, in_features) - initialized with small random values
        # B: (out_features, r) - initialized to zero (no change at start)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero-initialized
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Flag to track if weights are merged
        self.merged = False
    
    # Expose base layer attributes for compatibility
    @property
    def weight(self):
        """Expose weight from base layer for compatibility."""
        return self.base_layer.weight
    
    @property
    def bias(self):
        """Expose bias from base layer for compatibility."""
        return self.base_layer.bias
    
    @property
    def in_features(self):
        """Expose in_features from base layer."""
        return self.base_layer.in_features
    
    @property
    def out_features(self):
        """Expose out_features from base layer."""
        return self.base_layer.out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
        """
        # Base output from frozen weights
        base_output = self.base_layer(x)
        
        if self.merged:
            # Weights already merged, just use base layer
            return base_output
        
        # LoRA path: x -> dropout -> A -> B -> scale
        # Equivalent to: (x @ A.T) @ B.T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return base_output + lora_output * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights (B @ A) into base layer weights."""
        if self.merged:
            # Already merged, silently skip
            return
        
        with torch.no_grad():
            # Compute LoRA contribution: B @ A * scaling
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            # Add to base weight
            self.base_layer.weight.data += lora_weight
        
        self.merged = True
        logger.debug(f"Merged LoRA weights (r={self.r}) into base layer")
    
    def unmerge_weights(self):
        """Separate LoRA weights from base layer (reverse of merge)."""
        if not self.merged:
            # Not merged, silently skip
            return
        
        with torch.no_grad():
            # Subtract LoRA contribution
            lora_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= lora_weight
        
        self.merged = False
        logger.debug(f"Unmerged LoRA weights (r={self.r}) from base layer")
    
    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, merged={self.merged}"


# =============================================================================
# LoRA Application Functions
# =============================================================================

def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> Tuple[nn.Module, Dict[str, LoRALayer]]:
    """
    Apply LoRA to model with different rules for encoder vs non-encoder layers.
    
    For encoder: Only apply LoRA to target modules specified in config.target_modules.
    For non-encoder layers: Apply LoRA to ALL linear layers.
    
    Parameters
    ----------
    model : nn.Module
        The model to apply LoRA to (should have .encoder attribute for GLiNER2).
    config : LoRAConfig
        LoRA configuration.
    
    Returns
    -------
    model : nn.Module
        Modified model with LoRA layers.
    lora_layers : Dict[str, LoRALayer]
        Dictionary mapping layer names to LoRA layers.
    """
    if not config.enabled:
        logger.info("LoRA is disabled, skipping application")
        return model, {}
    
    lora_layers = {}
    encoder_lora_count = 0
    non_encoder_lora_count = 0
    
    def _should_apply_lora_to_encoder_module(module_name: str) -> bool:
        """Check if LoRA should be applied to this encoder module using substring matching."""
        return any(target in module_name for target in config.target_modules)
    
    # Recursively find and replace modules
    def _inject_lora_recursive(module: nn.Module, prefix: str = "", in_encoder: bool = False):
        nonlocal encoder_lora_count, non_encoder_lora_count
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if we're entering the encoder
            is_encoder_module = (name == "encoder" and not prefix) or in_encoder
            
            # Determine if we should apply LoRA to this Linear layer
            should_apply = False
            if isinstance(child, nn.Linear):
                if is_encoder_module:
                    # In encoder: only apply to target modules
                    should_apply = _should_apply_lora_to_encoder_module(name)
                else:
                    # Outside encoder: apply to all Linear layers
                    should_apply = True
            
            if should_apply:
                # Replace with LoRA layer
                lora_layer = LoRALayer(
                    base_layer=child,
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
                lora_layers[f"model.{full_name}"] = lora_layer
                
                if is_encoder_module:
                    encoder_lora_count += 1
                else:
                    non_encoder_lora_count += 1
                
                logger.debug(f"Applied LoRA to model.{full_name} (in={child.in_features}, out={child.out_features}, encoder={is_encoder_module})")
            else:
                # Recurse into child
                _inject_lora_recursive(child, full_name, is_encoder_module)
    
    _inject_lora_recursive(model)
    
    if not lora_layers:
        logger.warning(
            f"No LoRA layers were applied. For encoder, target modules {config.target_modules} "
            f"not found. Check your target_modules configuration."
        )
    else:
        logger.info(
            f"Applied LoRA to {len(lora_layers)} layers total: "
            f"{encoder_lora_count} in encoder (targeted), "
            f"{non_encoder_lora_count} in non-encoder (all linear)"
        )
    
    return model, lora_layers


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Extract all LoRA parameters (lora_A and lora_B) from model.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    List[nn.Parameter]
        List of LoRA parameters.
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get state dict containing only LoRA parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        State dict with LoRA parameters only.
    """
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data
    return lora_state


def merge_lora_weights(model: nn.Module) -> int:
    """
    Merge all LoRA weights into their base layers.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    int
        Number of layers merged.
    """
    count = 0
    already_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if not module.merged:
                module.merge_weights()
                count += 1
            else:
                already_merged += 1
    
    if count > 0:
        logger.debug(f"Merged LoRA weights in {count} layers")
    if already_merged > 0:
        logger.debug(f"Skipped {already_merged} layers (already merged)")
    return count


def unmerge_lora_weights(model: nn.Module) -> int:
    """
    Unmerge all LoRA weights from their base layers.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    int
        Number of layers unmerged.
    """
    count = 0
    not_merged = 0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if module.merged:
                module.unmerge_weights()
                count += 1
            else:
                not_merged += 1
    
    if count > 0:
        logger.debug(f"Unmerged LoRA weights in {count} layers")
    if not_merged > 0:
        logger.debug(f"Skipped {not_merged} layers (not merged)")
    return count


def count_lora_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count LoRA parameters vs total parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    lora_params : int
        Number of trainable LoRA parameters.
    total_params : int
        Total number of model parameters.
    percentage : float
        Percentage of trainable parameters.
    """
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    percentage = (lora_params / total_params * 100) if total_params > 0 else 0.0
    
    return lora_params, total_params, percentage


def print_lora_info(model: nn.Module, config: LoRAConfig):
    """
    Print detailed LoRA configuration and parameter statistics.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    config : LoRAConfig
        LoRA configuration.
    """
    lora_params, total_params, percentage = count_lora_parameters(model)
    
    # Count LoRA layers
    num_lora_layers = sum(1 for m in model.modules() if isinstance(m, LoRALayer))
    
    print("=" * 70)
    print("🔧 LoRA Configuration")
    print("=" * 70)
    print(f"Enabled            : {config.enabled}")
    print(f"Rank (r)           : {config.r}")
    print(f"Alpha              : {config.alpha}")
    print(f"Scaling (α/r)      : {config.alpha / config.r:.4f}")
    print(f"Dropout            : {config.dropout}")
    print(f"Target modules     : {', '.join(config.target_modules)}")
    print(f"LoRA layers        : {num_lora_layers}")
    print("-" * 70)
    print(f"Trainable params   : {lora_params:,} / {total_params:,} ({percentage:.2f}%)")
    print(f"Memory savings     : ~{100 - percentage:.1f}% fewer gradients")
    print("=" * 70)


def remove_lora_from_model(model: nn.Module) -> nn.Module:
    """
    Remove LoRA layers and restore original Linear layers.
    Useful for inference with merged weights.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
    
    Returns
    -------
    nn.Module
        Model with LoRA layers replaced by standard Linear layers.
    """
    def _remove_lora_recursive(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoRALayer):
                # Ensure weights are merged
                if not child.merged:
                    child.merge_weights()
                # Replace LoRALayer with its base layer
                setattr(module, name, child.base_layer)
                logger.debug(f"Removed LoRA from {name}, restored base layer")
            else:
                _remove_lora_recursive(child)
    
    _remove_lora_recursive(model)
    logger.info("Removed all LoRA layers from model")
    return model

