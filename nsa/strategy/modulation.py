from __future__ import annotations

from typing import Mapping, Sequence

from .config import StrategicConfig
from .model import StrategicField


class StrategicAttentionModulator:
    """Apply bounded strategic bias; it never changes authority or policy."""

    def __init__(self, max_bias=.25, *, enabled: bool = True):
        if max_bias <= 0:
            raise ValueError("max_bias must be positive")
        self.max_bias = float(max_bias)
        self.enabled = enabled

    def token_bias(self, field: StrategicField, affinity: Sequence[Mapping[str, float]]) -> tuple[float, ...]:
        if not self.enabled:
            return tuple(0.0 for _ in affinity)
        out = []
        for a in affinity:
            raw = sum(field.strategy_weights.get(k, 0.0) * v for k, v in a.items())
            out.append(max(-1.0, min(1.0, raw)) * self.max_bias * field.strength)
        return tuple(out)

    def apply(self, logits, field: StrategicField, affinity: Sequence[Mapping[str, float]]):
        bias = self.token_bias(field, affinity)
        if hasattr(logits, "new_tensor"):
            return logits + logits.new_tensor(bias)
        return tuple(float(v) + bias[i] for i, v in enumerate(logits))


def configured_modulator(config: StrategicConfig | None = None) -> StrategicAttentionModulator:
    cfg = config or StrategicConfig.from_env()
    return StrategicAttentionModulator(cfg.max_bias, enabled=cfg.enabled and cfg.attention_enabled)
