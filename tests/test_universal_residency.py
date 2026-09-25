from types import SimpleNamespace

import torch
from torch import nn

from nsa.residency.model_topology import profile_model_topology
from nsa.residency.streaming_executor import InMemoryRegionStore, StreamingExecutor


class TinyLayer(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.attention = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden * 2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer(), TinyLayer()])
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head


def test_profile_model_topology_is_ordered_and_bounded():
    plan = profile_model_topology(TinyModel())

    assert [r.name for r in plan.regions] == [
        "embeddings", "layer.0", "layer.1", "layer.2", "lm_head"
    ]
    assert plan.storage_bytes > 0
    assert plan.working_set_bytes == max(r.parameter_bytes for r in plan.regions)
    assert plan.regions[2].dependencies == ("layer.1",)


def test_streaming_executor_never_exceeds_region_budget():
    regions = {"a": object(), "b": object(), "c": object()}
    store = InMemoryRegionStore(regions, {"a": 10, "b": 20, "c": 15})
    executor = StreamingExecutor(store, budget_bytes=20)
    seen = []

    telemetry = executor.run(["a", "b", "c"], lambda name, _: seen.append(name))

    assert seen == ["a", "b", "c"]
    assert telemetry.bytes_loaded == 45
    assert telemetry.peak_resident_bytes == 20
    assert telemetry.regions_evicted == 3


def test_streaming_executor_rejects_oversized_region():
    store = InMemoryRegionStore({"large": object()}, {"large": 21})
    executor = StreamingExecutor(store, budget_bytes=20)

    try:
        executor.run(["large"], lambda *_: None)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected oversized region to fail")
