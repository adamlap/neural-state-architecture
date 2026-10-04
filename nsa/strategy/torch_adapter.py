"""Framework-light PyTorch attention integration for StrategicField."""

from __future__ import annotations

from typing import Mapping, Sequence

from .config import StrategicConfig
from .model import StrategicField
from .modulation import StrategicAttentionModulator


class StrategicAttentionBias:
    """Create a broadcastable key-position bias for attention logits."""

    def __init__(self, max_bias=.25, *, enabled: bool = True):
        self.modulator = StrategicAttentionModulator(max_bias, enabled=enabled)

    @classmethod
    def from_config(cls, config: StrategicConfig | None = None) -> "StrategicAttentionBias":
        cfg = config or StrategicConfig.from_env()
        return cls(cfg.max_bias, enabled=cfg.enabled and cfg.attention_enabled)

    def build(self, logits, field: StrategicField, token_affinity: Sequence[Mapping[str, float]]):
        values = self.modulator.token_bias(field, token_affinity)
        if not hasattr(logits, "new_tensor"):
            raise TypeError("logits must be a torch-like tensor")
        bias = logits.new_tensor(values)
        while bias.ndim < logits.ndim:
            bias = bias.unsqueeze(0)
        return bias

    def apply(self, logits, field: StrategicField, token_affinity):
        return logits + self.build(logits, field, token_affinity)
