from __future__ import annotations

import asyncio
import pytest

from nsa.residency.async_manager import AsyncResidencyManager
from nsa.residency.execution_backend import CpuBackend, DeviceInfo
from nsa.residency.execution_graph import ExecutionGraph, ExecutionGraphRunner, ExecutionOp


def test_topological_order_and_cycle_detection():
    graph = ExecutionGraph()
    graph.add(ExecutionOp("a", "identity", output="x"))
    graph.add(ExecutionOp("b", "identity", inputs=("x",), output="y", depends_on=("a",)))
    graph.add(ExecutionOp("c", "identity", depends_on=("b",)))
    assert [op.op_id for op in graph.topological_order()] == ["a", "b", "c"]

    cyclic = ExecutionGraph()
    cyclic.add(ExecutionOp("a", "identity", depends_on=("b",)))
    cyclic.add(ExecutionOp("b", "identity", depends_on=("a",)))
    with pytest.raises(ValueError, match="cycle"):
        cyclic.topological_order()


def test_unknown_dependency_is_rejected():
    graph = ExecutionGraph()
    graph.add(ExecutionOp("a", "identity", depends_on=("missing",)))
    with pytest.raises(ValueError, match="unknown"):
        graph.topological_order()


def test_runner_loads_only_required_regions():
    async def scenario():
        loads = []

        async def loader(name):
            loads.append(name)
            return name

        residency = AsyncResidencyManager(loader, lambda _: 1, capacity_bytes=4)
        graph = ExecutionGraph()
        graph.add(ExecutionOp("a", "identity", inputs=("x",), output="y", required_regions=("layer.0",)))
        runner = ExecutionGraphRunner(CpuBackend(), residency)
        result = await runner.run(graph, {"x": 7})
        assert result["y"] == 7
        assert loads == ["layer.0"]
        assert runner.metrics.region_requests == 1

    asyncio.run(scenario())


def test_runner_enforces_backend_capabilities():
    class Backend(CpuBackend):
        def device_info(self):
            return DeviceInfo("test", "custom", 1024, capabilities=frozenset({"matmul"}))

    async def scenario():
        residency = AsyncResidencyManager(lambda _: 1, lambda _: 1, capacity_bytes=4)
        graph = ExecutionGraph()
        graph.add(ExecutionOp("a", "identity", required_capabilities=frozenset({"tensorcore"})))
        with pytest.raises(RuntimeError, match="unsupported"):
            await ExecutionGraphRunner(Backend(), residency).run(graph)

    asyncio.run(scenario())
