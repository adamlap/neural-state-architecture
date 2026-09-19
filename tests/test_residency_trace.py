from nsa.residency import MemoryTier, ResidencyEvent, ResidencyTrace
from nsa.runtime.inference.model_registry import get_local_model


def test_trace_is_bounded_and_reports_metrics():
    trace = ResidencyTrace(max_events=2)
    trace.record(ResidencyEvent(1.0, "layer.0", "prefetch", MemoryTier.NVME, MemoryTier.RAM, 10, 2.0, "hit"))
    trace.record(ResidencyEvent(2.0, "layer.1", "execute", MemoryTier.RAM, MemoryTier.RAM, 0, 3.0, ""))
    trace.record(ResidencyEvent(3.0, "layer.2", "resident", MemoryTier.NVME, MemoryTier.VRAM, 20, 4.0, ""))
    metrics = trace.metrics()
    assert len(trace.events()) == 2
    assert metrics["prefetches"] == 0
    assert metrics["bytes_moved"] == 20


def test_model_registry_resolves_environment_override():
    spec = get_local_model("3B")
    assert spec.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert spec.resolve_path({"NSA_QWEN_3B_PATH": "/models/qwen3b"}) == "/models/qwen3b"
