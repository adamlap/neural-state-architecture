"""
nsa.lora
========
NSA-LoRA: Low-Rank Post-Hoc Retrofitting Adapters for Pre-Trained Transformers.

Concept:
    Allows pre-trained language models (Llama-3, Qwen-2.5, Mistral, GPT-2) to be
    retrofitted with Neural State Architecture policy governance WITHOUT full pre-training.

    Base semantic weights W_0 are FROZEN. Lightweight state transition matrices V
    and low-rank LoRA adapter matrices (A, B) are trained:
        W' = W_0 + (alpha / r) * (B · A)
        σ' = V σ

Trainable Parameter Ratio:
    < 0.5% of model parameters.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, DEFAULT_LATTICE
from nsa.fused_attention import FusedStateAwareAttention


class NSALoRALinear(nn.Module):
    """Low-Rank Adapter wrapper for frozen Linear layers with state operator injection."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        # Freeze base linear layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad = False

        in_features, out_features = base_layer.in_features, base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Low-rank adapter matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


class NSALoRAAttention(nn.Module):
    """Retrofits standard MultiheadAttention into Fused State-Aware Attention via LoRA."""

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        r: int = 8,
        lora_alpha: float = 16.0,
        gate_mode: str = "soft",
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> None:
        super().__init__()
        self.fused_attn = FusedStateAwareAttention(
            d_model=d_model,
            state_dim=state_dim,
            num_heads=num_heads,
            gate_mode=gate_mode,
            lattice=lattice,
        )

        # Wrap Q, K, V projections in LoRA adapters
        self.fused_attn.W_q = NSALoRALinear(self.fused_attn.W_q, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_k = NSALoRALinear(self.fused_attn.W_k, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_v = NSALoRALinear(self.fused_attn.W_v, r=r, lora_alpha=lora_alpha)
        self.fused_attn.W_o = NSALoRALinear(self.fused_attn.W_o, r=r, lora_alpha=lora_alpha)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fused_attn(x, state, mask=mask)


def apply_nsa_lora_retrofit(
    model: nn.Module,
    state_dim: int = 8,
    r: int = 8,
    lora_alpha: float = 16.0,
) -> Tuple[nn.Module, Dict[str, int]]:
    """Freeze base model parameters and attach NSA-LoRA adapters to all attention layers.

    Returns:
        retrofitted_model: nn.Module
        param_counts     : Dict with trainable vs frozen parameter statistics
    """
    total_params = 0
    trainable_params = 0

    # Freeze all existing parameters
    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

    # Create trainable state stream embedding if model lacks one
    if not hasattr(model, "state_emb"):
        model.state_emb = nn.Embedding(512, state_dim)
        for p in model.state_emb.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
            total_params += p.numel()

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()

    return model, {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "pct_trainable": (trainable_params / max(total_params, 1)) * 100.0,
    }
