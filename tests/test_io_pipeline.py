"""Tests for asynchronous materialization pipeline."""
import torch
from nsa.residency.io_pipeline import AsyncMaterializationPipeline, PipelineTask
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyPolicy
from nsa.residency.quantized_storage import QuantizedRegionStore
from nsa.residency.types import MemoryTier, NeuralRegion


def test_pipeline_execution():
    policy = ResidencyPolicy(vram_budget_bytes=10*1024*1024, ram_budget_bytes=20*1024*1024)
    manager = NeuralResidencyManager(policy)
    region = NeuralRegion("layer.0.expert.2", size_bytes=1024)
    manager.register([region])

    q_store = QuantizedRegionStore(target_precision="int8")
    tensors = {"weight": torch.randn(16, 16)}
    q_store.store_region("layer.0.expert.2", tensors)

    pipeline = AsyncMaterializationPipeline(
        manager=manager,
        quantized_store=q_store,
        max_io_workers=1,
        max_dequant_workers=1,
    )

    task = PipelineTask(
        region_id="layer.0.expert.2",
        target_tier=MemoryTier.RAM,
        precision="int8",
        priority=1.0,
    )

    scheduled = pipeline.schedule_materialization(task)
    assert scheduled is True

    pipeline.wait("layer.0.expert.2", timeout=2.0)
    assert manager.states["layer.0.expert.2"].value == "resident"
    assert manager.tiers["layer.0.expert.2"] == MemoryTier.RAM

    staged = pipeline.get_staged_tensors("layer.0.expert.2")
    assert staged is not None
    assert "weight" in staged
    assert staged["weight"].shape == (16, 16)

    # Events should include pipeline-io-start, pipeline-io-complete, pipeline-dequant-complete, resident
    actions = [e.action for e in manager.events]
    assert "pipeline-io-start" in actions
    assert "pipeline-io-complete" in actions
    assert "pipeline-dequant-complete" in actions
    assert "resident" in actions

    pipeline.shutdown(wait=True)