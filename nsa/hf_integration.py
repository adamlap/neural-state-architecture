"""
nsa.hf_integration
==================
Lightweight HuggingFace-style wrappers and retrofitting utilities for NSA models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from nsa.layers import NSACausalLM
from nsa.lora import NSALoRALinear

try:
    from transformers import PretrainedConfig

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PretrainedConfig = object  # type: ignore


if HAS_TRANSFORMERS:

    class NSAConfig(PretrainedConfig):
        """HF PretrainedConfig subclass when transformers is available."""

        model_type = "nsa"

        def __init__(
            self,
            vocab_size: int = 32000,
            d_model: int = 512,
            state_dim: int = 8,
            num_layers: int = 8,
            num_heads: int = 8,
            compat_mode: str = "level",
            gate_mode: str = "hard",
            dropout: float = 0.0,
            max_seq_len: int = 2048,
            **kwargs: Any,
        ):
            super().__init__(**kwargs)
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.hidden_size = d_model
            self.state_dim = state_dim
            self.num_layers = num_layers
            self.num_hidden_layers = num_layers
            self.num_heads = num_heads
            self.num_attention_heads = num_heads
            self.compat_mode = compat_mode
            self.gate_mode = gate_mode
            self.dropout = dropout
            self.max_seq_len = max_seq_len
else:

    class NSAConfig:
        """Minimal config object when transformers is not installed."""

        model_type = "nsa"

        def __init__(
            self,
            vocab_size: int = 32000,
            d_model: int = 512,
            state_dim: int = 8,
            num_layers: int = 8,
            num_heads: int = 8,
            compat_mode: str = "level",
            gate_mode: str = "hard",
            dropout: float = 0.0,
            max_seq_len: int = 2048,
            **kwargs: Any,
        ):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.hidden_size = d_model
            self.state_dim = state_dim
            self.num_layers = num_layers
            self.num_hidden_layers = num_layers
            self.num_heads = num_heads
            self.num_attention_heads = num_heads
            self.compat_mode = compat_mode
            self.gate_mode = gate_mode
            self.dropout = dropout
            self.max_seq_len = max_seq_len
            for k, v in kwargs.items():
                setattr(self, k, v)

        def to_dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)


class NSAForCausalLM(nn.Module):
    """Causal LM wrapper with HF-like forward dict outputs."""

    config_class = NSAConfig

    def __init__(self, config: NSAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NSACausalLM(
            vocab_size=config.vocab_size,
            d_model=getattr(config, "d_model", getattr(config, "hidden_size", 512)),
            state_dim=config.state_dim,
            num_layers=getattr(config, "num_layers", getattr(config, "num_hidden_layers", 8)),
            num_heads=getattr(config, "num_heads", getattr(config, "num_attention_heads", 8)),
            max_seq_len=getattr(config, "max_seq_len", 2048),
            compat_mode=getattr(config, "compat_mode", "level"),
            gate_mode=getattr(config, "gate_mode", "hard"),
            dropout=getattr(config, "dropout", 0.0),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        state_vectors: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[torch.Tensor]]:
        del attention_mask, kwargs
        logits, _, state_out = self.model(input_ids, state_init=state_vectors)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "state_vectors": state_out,
        }

    @classmethod
    def from_config(cls, config: NSAConfig) -> NSAForCausalLM:
        return cls(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.nsa.tok_emb

    def get_output_embeddings(self) -> nn.Module:
        return self.model.lm_head


def retrofit_llama_attention(model: nn.Module, r: int = 8) -> Tuple[nn.Module, int, int]:
    """Freeze all original parameters and insert ``NSALoRALinear`` adapters into attention blocks."""
    frozen_params = 0
    trainable_params = 0

    for p in model.parameters():
        p.requires_grad = False
        frozen_params += p.numel()

    # Replace linear projections inside each attention block
    for _name, module in model.named_modules():
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "W_q", "W_k", "W_v", "W_o"]:
            if hasattr(module, proj_name):
                old = getattr(module, proj_name)
                if isinstance(old, nn.Linear):
                    new = NSALoRALinear(old, r=r)
                    setattr(module, proj_name, new)
                    for p in new.parameters():
                        if p.requires_grad:
                            trainable_params += p.numel()

    return model, frozen_params, trainable_params


# Generic alias
retrofit_hf_attention = retrofit_llama_attention
