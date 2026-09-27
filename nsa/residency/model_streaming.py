"""End-to-end execution of a compiled model graph under bounded residency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from .execution_backend import DeviceInfo, ExecutionBackend
from .model_plan import ModelResidencyPlan
from .prefetch import SequentialPrefetcher


@dataclass
class StreamingRunMetrics:
    operations: int = 0
    region_requests: int = 0
    operation_seconds: float = 0.0
    region_seconds: float = 0.0
    explicit_evictions: int = 0
    device_moves: int = 0
    device_releases: int = 0
    prefetch_decisions: int = 0


class ModelStreamingExecutor:
    """Run a model plan while keeping only the graph working set resident."""

    def __init__(
        self,
        plan: ModelResidencyPlan,
        backend: ExecutionBackend,
        residency: Any,
        device: DeviceInfo | None = None,
        prefetcher: SequentialPrefetcher | None = None,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.residency = residency
        self.device = device or backend.device_info()
        self.prefetcher = prefetcher
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

    async def _prefetch_next(self, operations: tuple[Any, ...], index: int) -> None:
        if self.prefetcher is None or index + 1 >= len(operations):
            return
        current = operations[index]
        next_op = operations[index + 1]
        candidates = list(next_op.required_regions)
        previous = current.required_regions[-1] if current.required_regions else current.op_id
        decisions = self.prefetcher.predict(previous, candidates, top_k=len(candidates))
        self.metrics.prefetch_decisions += len(decisions)
        if decisions:
            for decision in decisions:
                self.prefetcher.observe(previous, decision.region)
            await self.residency.prefetch([decision.region for decision in decisions])

    async def run(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = dict(values or {})
        operations = self.plan.execution_graph.topological_order()
        remaining = self._remaining_uses(operations)

        for index, op in enumerate(operations):
            self._check_capabilities(op)
            started = perf_counter()
            moved_regions: dict[str, Any] = {}
            if op.required_regions:
                region_started = perf_counter()
                loaded = await asyncio.gather(
                    *(self.residency.get(region) for region in op.required_regions)
                )
                self.metrics.region_seconds += perf_counter() - region_started
                self.metrics.region_requests += len(op.required_regions)
                for region, value in zip(op.required_regions, loaded):
                    moved_regions[region] = self.backend.move(value, self.device)
                    self.metrics.device_moves += 1

            inputs = tuple(state[name] for name in op.inputs)
            result = self.backend.execute(
                op.operation,
                inputs,
                regions=moved_regions,
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
                    self.backend.release(moved_regions[region])
                    self.metrics.device_releases += 1

            await self._prefetch_next(operations, index)

        drain = getattr(self.residency, "drain_prefetches", None)
        if callable(drain):
            await drain()
        else:
            inner = getattr(self.residency, "residency", None)
            drain_inner = getattr(inner, "drain_prefetches", None)
            if callable(drain_inner):
                await drain_inner()
        return state
