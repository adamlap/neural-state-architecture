"""Unified model residency planning artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .execution_graph import ExecutionGraph
from .model_execution import compile_execution_graph, compile_residency_regions
from .types import NeuralRegion
from .weight_index import WeightIndex


@dataclass(frozen=True)
class ModelResidencyPlan:
    regions: tuple[NeuralRegion, ...]
    execution_graph: ExecutionGraph
    weight_index: WeightIndex | None = None

    @property
    def storage_bytes(self) -> int:
        return sum(region.size_bytes for region in self.regions)

    @property
    def persistent_state_bytes(self) -> int:
        return sum(
            region.size_bytes
            for region in self.regions
            if "persistent_state" in region.semantic_tags or "kv_state" in region.semantic_tags
        )

    @property
    def peak_required_region_bytes(self) -> int:
        sizes = {region.region_id: region.size_bytes for region in self.regions}
        return max(
            (sum(sizes.get(region, 0) for region in op.required_regions)
             for op in self.execution_graph.operations),
            default=0,
        )

    @property
    def region_count(self) -> int:
        return len(self.regions)


def build_model_residency_plan(
    config: Mapping[str, Any],
    *,
    weight_index: WeightIndex | None = None,
    region_size_bytes: Mapping[str, int] | None = None,
) -> ModelResidencyPlan:
    """Compile topology, exact weight metadata and execution into one plan."""
    sizes = dict(region_size_bytes or {})
    if weight_index is not None:
        for region, size in weight_index.region_sizes().items():
            sizes.setdefault(region, size)
    regions = compile_residency_regions(config, region_size_bytes=sizes)
    graph = compile_execution_graph(config)
    return ModelResidencyPlan(regions, graph, weight_index)
