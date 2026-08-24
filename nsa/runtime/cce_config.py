"""Configuration schema and loader for Continuous Cognitive Engine (Phase CCE-8)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class CCEConfig:
    """Typed configuration for CCE runtime."""

    dimension: int = 4
    decay_rate: float = 0.05
    learning_rate: float = 0.4
    tick_interval_seconds: float = 1.0
    feedback_max_norm: float = 0.25
    host: str = "0.0.0.0"
    port: int = 8000
    model: str = "qwen2.5:3b"
    backend: str = "ollama"
    checkpoint_dir: str = ".cce_checkpoints"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CCEConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CCEConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


__all__ = ["CCEConfig"]
