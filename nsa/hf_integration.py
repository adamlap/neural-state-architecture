"""
nsa.hf_integration
==================
HuggingFace `transformers` Ecosystem Integration for NSA.

Enables seamless compatibility with standard HuggingFace interfaces:
    - NSAConfig (PretrainedConfig compatible)
    - NSAForCausalLM (AutoModelForCausalLM compatible)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from nsa.layers import NSACausalLM
from nsa.algebra import StateLattice, DEFAULT_LATTICE


class NSAConfig:
    """Configuration class for Neural State Architecture HuggingFace integration."""

    model_type = "nsa"

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        state_dim: int = 8,
        num_layers: int = 32,
        num_heads: int = 32,
        compat_mode: str = "dot",
        gate_mode: str = "soft",
        dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.compat_mode = compat_mode
        self.gate_mode = gate_mode
        self.dropout = dropout
        for k, v in kwargs.items():
            setattr(self, k, v)


class NSAForCausalLM(nn.Module):
    """HuggingFace AutoModelForCausalLM compatible model wrapper for NSA."""

    def __init__(self, config: NSAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NSACausalLM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            compat_mode=config.compat_mode,
            gate_mode=config.gate_mode,
            dropout=config.dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        state_vectors: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits, _, state_out = self.model(input_ids, state_init=state_vectors)

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "state_vectors": state_out,
        }

    @classmethod
    def from_config(cls, config: NSAConfig) -> NSAForCausalLM:
        return cls(config)
