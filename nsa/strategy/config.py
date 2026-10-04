from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class StrategicConfig:
    """Runtime configuration for the optional SNM control layer."""
    enabled: bool = True
    feedback_enabled: bool = True
    attention_enabled: bool = True
    max_bias: float = 0.25

    def __post_init__(self) -> None:
        if self.max_bias <= 0.0:
            raise ValueError("max_bias must be positive")

    @classmethod
    def from_env(cls, prefix: str = "NSA_SNM_") -> "StrategicConfig":
        raw_max_bias = os.getenv(f"{prefix}MAX_BIAS")
        return cls(
            enabled=_env_bool(f"{prefix}ENABLED", True),
            feedback_enabled=_env_bool(f"{prefix}FEEDBACK_ENABLED", True),
            attention_enabled=_env_bool(f"{prefix}ATTENTION_ENABLED", True),
            max_bias=float(raw_max_bias) if raw_max_bias is not None else 0.25,
        )
