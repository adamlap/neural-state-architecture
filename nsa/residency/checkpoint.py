"""Build a residency plan from checkpoint metadata without loading weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_plan import ModelResidencyPlan, build_model_residency_plan
from .safetensors_metadata import inspect_safetensors_metadata
from .weight_index import load_safetensors_index


def load_checkpoint_plan(
    root: str | Path,
    *,
    config_name: str = "config.json",
    index_name: str = "model.safetensors.index.json",
) -> ModelResidencyPlan:
    """Inspect config and Safetensors headers, then compile one model plan."""
    root_path = Path(root)
    config: dict[str, Any] = json.loads(
        (root_path / config_name).read_text(encoding="utf-8")
    )
    index = load_safetensors_index(root_path / index_name)
    inspected = inspect_safetensors_metadata(root_path, index)
    return build_model_residency_plan(config, weight_index=inspected)
