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
        if configured:
            return str(Path(configured).expanduser())
        # Honour the Hugging Face cache overrides for the default cache location.
        hub_prefix = "~/.cache/huggingface/hub/"
        if self.default_path.startswith(hub_prefix):
            hub = values.get("HF_HUB_CACHE") or (
                str(Path(values["HF_HOME"]) / "hub") if values.get("HF_HOME") else None
            )
            if hub:
                return str(Path(hub).expanduser() / self.default_path[len(hub_prefix):])
        return str(Path(self.default_path).expanduser())

    def is_local(self, environ: Mapping[str, str] | None = None) -> bool:
        """True when the resolved location holds a usable checkpoint (has a config.json)."""
        return (Path(self.checkpoint_path(environ)) / "config.json").is_file()

    def checkpoint_path(self, environ: Mapping[str, str] | None = None) -> str:
        """Resolve a Hugging Face cache dir to its active snapshot directory.

        Prefers the snapshot named by ``refs/main``, then the most recently
        modified complete snapshot (one containing ``config.json``), then any
        snapshot. A plain checkpoint directory is returned unchanged.
        """
        path = Path(self.resolve_path(environ))
        snapshots = path / "snapshots"
        if not snapshots.is_dir():
            return str(path)
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
        if not candidates:
            return str(path)
        ref = path / "refs" / "main"
        if ref.is_file():
            pinned = snapshots / ref.read_text(encoding="utf-8").strip()
            if pinned.is_dir() and (pinned / "config.json").is_file():
                return str(pinned)
        complete = [p for p in candidates if (p / "config.json").is_file()]
        pool = complete or candidates
        return str(max(pool, key=lambda p: (p.stat().st_mtime, p.name)))


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
