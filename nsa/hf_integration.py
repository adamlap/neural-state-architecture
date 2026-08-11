"""
nsa.hf_integration
==================
Lightweight HuggingFace-style wrappers for NSA models.

This is a *prototype* compatibility layer:
  - NSAConfig mirrors common HF config fields (not a real PretrainedConfig subclass
    unless `transformers` is installed).
  - NSAForCausalLM exposes a dict-style forward used by simple training loops.

It does **not** register with AutoModel or provide full `generate()` / cache
parity with production HF models.  For real HF model retrofit + generate(),
see ``prototype/llama_security_showcase.py`` (mask injection hooks).
"""

from __future__ import annotations

from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from nsa.layers import NSACausalLM

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
    def from_config(cls, config: NSAConfig) -> "NSAForCausalLM":
        return cls(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.nsa.tok_emb

    def get_output_embeddings(self) -> nn.Module:
        return self.model.lm_head
