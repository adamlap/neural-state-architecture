import pytest

from nsa.residency.execution_backend import DeviceInfo
from nsa.residency.model_plan import build_model_residency_plan
from nsa.residency.model_streaming import ModelStreamingExecutor
from nsa.residency.tiered_store import TieredRegionStore


class FakeBackend:
    def device_info(self):
        return DeviceInfo("fake", "cpu", 7, 1, frozenset())

    def move(self, value, device):
        return value

    def release(self, value):
        pass

    def execute(self, operation, inputs, *, regions=None, persistent_regions=(), metadata=None):
        if operation == "embedding":
            return 1
        if operation in {"linear_attention", "full_attention", "transformer"}:
            return inputs[0] + 1
        if operation == "lm_head":
            return inputs[0] * 10
        raise ValueError(operation)


class FakeSource:
    def __init__(self, sizes):
        self.sizes = sizes
        self.loads = []

    def load_region(self, region):
        self.loads.append(region)
        return region

    def size_of(self, region):
        return self.sizes[region]


@pytest.mark.asyncio
async def test_model_streaming_respects_bounded_hot_memory():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 3,
            "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        },
    }
    sizes = {
        "embeddings": 2,
        "layer.0": 4,
        "layer.1": 4,
        "layer.2": 4,
        "lm_head": 3,
        "model_state": 0,
    }
    plan = build_model_residency_plan(config, region_size_bytes=sizes)
    source = FakeSource({
        "embeddings": 2,
        "layer.0.weights": 4,
        "layer.0.state": 0,
        "layer.1.weights": 4,
        "layer.1.state": 0,
        "layer.2.weights": 4,
        "layer.2.state": 0,
        "model_state": 0,
        "lm_head": 3,
    })
    store = TieredRegionStore(source, capacity_bytes=7)
    runner = ModelStreamingExecutor(plan, FakeBackend(), store)

    result = await runner.run()

    assert result["logits"] == 40
    assert store.resident_bytes <= 7
    assert runner.metrics.operations == 5
    assert runner.metrics.device_moves == 9
    assert runner.metrics.explicit_evictions >= 4
    assert source.loads.count("layer.0.weights") == 1
    assert source.loads.count("layer.1.weights") == 1
    assert source.loads.count("layer.2.weights") == 1


@pytest.mark.asyncio
async def test_model_streaming_rejects_budget_smaller_than_region():
    config = {
        "text_config": {
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
        }
    }
    plan = build_model_residency_plan(
        config,
        region_size_bytes={"embeddings": 2, "layer.0": 8, "model_state": 0, "lm_head": 1},
    )
    source = FakeSource({
        "embeddings": 2,
        "layer.0.weights": 8,
        "layer.0.state": 0,
        "model_state": 0,
        "lm_head": 1,
    })
    store = TieredRegionStore(source, capacity_bytes=4)

    with pytest.raises(MemoryError, match="requires 8 bytes"):
        await ModelStreamingExecutor(plan, FakeBackend(), store).run()
