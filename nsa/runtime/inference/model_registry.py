"""Local model registry for NSA's residency-aware inference backends."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class LocalModelSpec:
    key: str
    model_id: str
    env_var: str
    default_path: str
    vram_budget_gb: float
    ram_budget_gb: float

    def resolve_path(self, environ: Mapping[str, str] | None = None) -> str:
        values = environ if environ is not None else os.environ
        configured = values.get(self.env_var)
        return str(Path(configured or self.default_path).expanduser())

    def is_local(self, environ: Mapping[str, str] | None = None) -> bool:
        return Path(self.resolve_path(environ)).exists()

    def checkpoint_path(self, environ: Mapping[str, str] | None = None) -> str:
        path = Path(self.resolve_path(environ))
        snapshots = path / "snapshots"
        if snapshots.is_dir():
            candidates = sorted(p for p in snapshots.iterdir() if p.is_dir())
            if candidates:
                return str(candidates[-1])
        return str(path)


LOCAL_MODELS: dict[str, LocalModelSpec] = {
    "1.5b": LocalModelSpec(
        "1.5b", "Qwen/Qwen2.5-1.5B-Instruct", "NSA_QWEN_1_5B_PATH",
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct",
        3.0, 6.0,
    ),
    "3b": LocalModelSpec(
        "3b", "Qwen/Qwen2.5-3B-Instruct", "NSA_QWEN_3B_PATH",
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct",
        4.0, 8.0,
    ),
}


def get_local_model(key: str) -> LocalModelSpec:
    try:
        return LOCAL_MODELS[key.lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown local NSA model {key!r}; choose one of {sorted(LOCAL_MODELS)}") from exc
