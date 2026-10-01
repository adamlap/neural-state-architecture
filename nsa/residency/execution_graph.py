"""Hardware-agnostic execution graph and scheduler."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping
from .execution_backend import DeviceInfo, ExecutionBackend

@dataclass(frozen=True)
class ExecutionOp:
    op_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    output: str | None = None
    required_regions: tuple[str, ...] = ()
    persistent_regions: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionMetrics:
    operations: int = 0
    operation_seconds: float = 0.0
    region_requests: int = 0
    region_seconds: float = 0.0

class ExecutionGraph:
    """A deterministic DAG of backend operations."""
    def __init__(self) -> None:
        self._ops: dict[str, ExecutionOp] = {}

    @property
    def operations(self) -> tuple[ExecutionOp, ...]:
        return tuple(self._ops.values())

    def add(self, op: ExecutionOp) -> None:
        if op.op_id in self._ops:
            raise ValueError(f"duplicate operation id {op.op_id!r}")
        self._ops[op.op_id] = op

    def topological_order(self) -> tuple[ExecutionOp, ...]:
        indegree = {op_id: 0 for op_id in self._ops}
        children: dict[str, list[str]] = {op_id: [] for op_id in self._ops}
        for op in self._ops.values():
            for dep in op.depends_on:
                if dep not in self._ops:
                    raise ValueError(f"operation {op.op_id!r} depends on unknown {dep!r}")
                indegree[op.op_id] += 1
                children[dep].append(op.op_id)
        ready = [op_id for op_id, degree in indegree.items() if degree == 0]
        ordered: list[ExecutionOp] = []
        while ready:
            op_id = ready.pop(0)
            ordered.append(self._ops[op_id])
            for child in children[op_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self._ops):
            raise ValueError("execution graph contains a dependency cycle")
        return tuple(ordered)

class ExecutionGraphRunner:
    """Execute operations while obtaining required regions on demand."""
    def __init__(self, backend: ExecutionBackend, residency: Any, device: DeviceInfo | None = None) -> None:
        self.backend = backend
        self.residency = residency
        self.device = device or backend.device_info()
        self.metrics = ExecutionMetrics()

    def _check_capabilities(self, op: ExecutionOp) -> None:
        missing = op.required_capabilities - self.device.capabilities
        if missing:
            raise RuntimeError(
                f"operation {op.op_id!r} requires unsupported capabilities: "
                + ", ".join(sorted(missing))
            )

    async def run(self, graph: ExecutionGraph, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        state = dict(values or {})
        for op in graph.topological_order():
            self._check_capabilities(op)
            started = perf_counter()
            if op.required_regions:
                region_started = perf_counter()
                loaded = await asyncio.gather(*(self.residency.get(region) for region in op.required_regions))
                self.metrics.region_seconds += perf_counter() - region_started
                self.metrics.region_requests += len(op.required_regions)
                regions = dict(zip(op.required_regions, loaded))
            else:
                regions = {}
            inputs = tuple(state[name] for name in op.inputs)
            result = self.backend.execute(
                op.operation, inputs, regions=regions,
                persistent_regions=op.persistent_regions, metadata=dict(op.metadata)
            )
            self.metrics.operations += 1
            self.metrics.operation_seconds += perf_counter() - started
            if op.output is not None:
                state[op.output] = result
        return state
