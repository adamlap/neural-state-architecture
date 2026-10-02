"""Bounded token-flow modulation for strategic control."""
from __future__ import annotations
from typing import Mapping, Sequence
from .model import StrategicField

class StrategicAttentionModulator:
    """Convert a StrategicField into bounded additive attention bias."""
    def __init__(self, *, max_bias: float = 0.25):
        if max_bias <= 0:
            raise ValueError("max_bias must be positive")
        self.max_bias = float(max_bias)

    def token_bias(self, field: StrategicField,
                   token_affinity: Sequence[Mapping[str, float]]) -> tuple[float, ...]:
        result = []
        for affinities in token_affinity:
            raw = sum(field.strategy_weights.get(strategy, 0.0) * affinity
                      for strategy, affinity in affinities.items())
            raw = max(-1.0, min(1.0, raw))
            result.append(raw * self.max_bias * field.strength)
        return tuple(result)

    def apply(self, attention_logits, *, field: StrategicField,
              token_affinity: Sequence[Mapping[str, float]]):
        """Add the bounded bias to a sequence or tensor-like attention logits."""
        bias = self.token_bias(field, token_affinity)
        if hasattr(attention_logits, "new_tensor"):
            return attention_logits + attention_logits.new_tensor(bias)
        try:
            return attention_logits + bias
        except TypeError:
            return tuple(float(v) + bias[i] for i, v in enumerate(attention_logits))
