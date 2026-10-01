"""Static model residency analysis and hardware-budget simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model_plan import ModelResidencyPlan


@dataclass(frozen=True)
class ResidencyAnalysis:
    storage_bytes: int
    persistent_state_bytes: int
    peak_required_bytes: int
    region_count: int
    operation_count: int
    fits_peak_budget: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage_bytes": self.storage_bytes,
            "persistent_state_bytes": self.persistent_state_bytes,
            "peak_required_bytes": self.peak_required_bytes,
            "region_count": self.region_count,
            "operation_count": self.operation_count,
            "fits_peak_budget": self.fits_peak_budget,
        }


def analyze_plan(
    plan: ModelResidencyPlan,
    *,
    residency_budget_bytes: int | None = None,
) -> ResidencyAnalysis:
    peak = plan.peak_required_region_bytes
    return ResidencyAnalysis(
        storage_bytes=plan.storage_bytes,
        persistent_state_bytes=plan.persistent_state_bytes,
        peak_required_bytes=peak,
        region_count=plan.region_count,
        operation_count=len(plan.execution_graph.operations),
        fits_peak_budget=(
            residency_budget_bytes is None or peak <= residency_budget_bytes
        ),
    )


def _layer_types_for_summary(text: Mapping[str, Any]) -> list[str]:
    explicit = text.get("layer_types")
    if explicit:
        return [str(x) for x in explicit]

    count = int(text.get("num_hidden_layers", 0))
    interval = int(text.get("full_attention_interval", 0))
    if interval > 0:
        return [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(count)
        ]
    return ["transformer"] * count


def qwen3_5_residency_summary(
    config: Mapping[str, Any],
    plan: ModelResidencyPlan,
) -> dict[str, Any]:
    """Return architecture facts without making Qwen-specific runtime decisions."""
    text = config.get("text_config", config)
    layer_types = _layer_types_for_summary(text)
    layers = int(text.get("num_hidden_layers", len(layer_types)))
    linear = sum(
        1 for x in layer_types
        if "linear" in str(x).lower() or "deltanet" in str(x).lower()
    )
    full = sum(
        1 for x in layer_types
        if "full" in str(x).lower()
        or ("attention" in str(x).lower() and "linear" not in str(x).lower() and "deltanet" not in str(x).lower())
    )
    return {
        "model_type": config.get("model_type", text.get("model_type", "")),
        "layers": layers,
        "linear_attention_layers": linear,
        "full_attention_layers": full,
        "hidden_size": int(text.get("hidden_size", 0)),
        "vocab_size": int(text.get("vocab_size", 0)),
        "context_length": int(text.get("max_position_embeddings", 0)),
        "has_vision": "vision_config" in config,
        "has_mtp": int(text.get("mtp_num_hidden_layers", 0)) > 0,
        "storage_bytes": plan.storage_bytes,
        "persistent_state_bytes": plan.persistent_state_bytes,
    }
