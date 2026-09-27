"""End-to-end execution of a compiled model graph under bounded residency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from .execution_backend import DeviceInfo, ExecutionBackend
from .model_plan import ModelResidencyPlan


@dataclass
class StreamingRunMetrics:
    operations: int = 0
    region_requests: int = 0
    operation_seconds: float = 0.0
    region_seconds: float = 0.0
    explicit_evictions: int = 0


class ModelStreamingExecutor:
    """Run a model plan while keeping only the graph working set resident.

    The executor deliberately knows nothing about CPU/GPU/NPU mechanics. It
    delegates computation to an ExecutionBackend and residency to a bounded
    store. Non-persistent regions are released after their last use, allowing
    the same model plan to run under different hardware memory budgets.
    """

    def __init__(
        self,
        plan: ModelResidencyPlan,
        backend: ExecutionBackend,
        residency: Any,
        device: DeviceInfo | None = None,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.residency = residency
        self.device = device or backend.device_info()
        self.metrics = StreamingRunMetrics()

    def _check_capabilities(self, op: Any) -> None:
        missing = op.required_capabilities - self.device.capabilities
        if missing:
            raise RuntimeError(
                f"operation {op.op_id!r} requires unsupported capabilities: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _remaining_uses(operations: tuple[Any, ...]) -> dict[str, int]:
        uses: dict[str, int] = {}
        for op in operations:
            for region in op.required_regions:
                uses[region] = uses.get(region, 0) + 1
        return uses

    async def run(
        self,
        values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = dict(values or {})
        operations = self.plan.execution_graph.topological_order()
        remaining = self._remaining_uses(operations)

        for op in operations:
            self._check_capabilities(op)
            started = perf_counter()
            if op.required_regions:
                region_started = perf_counter()
                loaded = await asyncio.gather(
                    *(self.residency.get(region) for region in op.required_regions)
                )
                self.metrics.region_seconds += perf_counter() - region_started
                self.metrics.region_requests += len(op.required_regions)
                regions = dict(zip(op.required_regions, loaded))
            else:
                regions = {}

            inputs = tuple(state[name] for name in op.inputs)
            result = self.backend.execute(
                op.operation,
                inputs,
                regions=regions,
                persistent_regions=op.persistent_regions,
                metadata=dict(op.metadata),
            )
            self.metrics.operations += 1
            self.metrics.operation_seconds += perf_counter() - started

            if op.output is not None:
                state[op.output] = result

            for region in op.required_regions:
                remaining[region] -= 1
                if remaining[region] == 0 and region not in op.persistent_regions:
                    if await self.residency.evict(region):
                        self.metrics.explicit_evictions += 1

        return state
