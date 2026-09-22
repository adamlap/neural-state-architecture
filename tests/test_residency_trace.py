import json

from nsa.residency import MemoryTier, ResidencyEvent, ResidencyTrace
from nsa.runtime.inference.model_registry import get_local_model


def _ev(ts, region, action, bytes_moved=0, latency=0.0, src=MemoryTier.NVME, dst=MemoryTier.RAM):
    return ResidencyEvent(ts, region, action, src, dst, bytes_moved, latency, "")


def test_trace_is_bounded_and_reports_metrics():
    trace = ResidencyTrace(max_events=2)
    trace.record(ResidencyEvent(1.0, "layer.0", "prefetch", MemoryTier.NVME, MemoryTier.RAM, 10, 2.0, "hit"))
    trace.record(ResidencyEvent(2.0, "layer.1", "execute", MemoryTier.RAM, MemoryTier.RAM, 0, 3.0, ""))
    trace.record(ResidencyEvent(3.0, "layer.2", "resident", MemoryTier.NVME, MemoryTier.VRAM, 20, 4.0, ""))
    metrics = trace.metrics()
    assert len(trace.events()) == 2
    assert metrics["prefetches"] == 0
    assert metrics["bytes_moved"] == 20


def test_executions_without_any_prefetch_are_never_hits():
    trace = ResidencyTrace()
    for i in range(10):
        trace.record(_ev(float(i), f"layer.{i}", "execute", latency=1.0))
    trace.record(_ev(20.0, "layer.0", "prefetch"))  # issued, never completed
    metrics = trace.metrics()
    assert metrics["prefetch_hits"] == 0
    assert metrics["prefetch_hit_rate"] == 0.0
    assert metrics["prefetch_coverage"] == 0.0


def test_hit_requires_completion_before_execution_starts():
    trace = ResidencyTrace()
    trace.record(_ev(1.0, "layer.1", "prefetch"))
    trace.record(_ev(5.0, "layer.1", "prefetch-complete", bytes_moved=100))
    # execute stamped at t=5.5 having run for 1000 ms -> started at 4.5, before the prefetch finished
    trace.record(_ev(5.5, "layer.1", "execute", latency=1000.0))
    assert trace.metrics()["prefetch_hits"] == 0
    # a second execution that starts after completion is not served either: the prefetch was consumed/never valid
    trace.record(_ev(9.0, "layer.1", "prefetch"))
    trace.record(_ev(9.5, "layer.1", "prefetch-complete", bytes_moved=100))
    trace.record(_ev(11.0, "layer.1", "execute", latency=500.0))  # starts at 10.5 > 9.5
    assert trace.metrics()["prefetch_hits"] == 1


def test_each_prefetch_serves_at_most_one_execution_and_rate_is_bounded():
    trace = ResidencyTrace()
    trace.record(_ev(1.0, "a", "prefetch"))
    trace.record(_ev(2.0, "a", "prefetch-complete", bytes_moved=64))
    for t in (3.0, 4.0, 5.0):
        trace.record(_ev(t, "a", "execute", latency=1.0))
    metrics = trace.metrics()
    assert metrics["prefetch_hits"] == 1
    assert metrics["prefetch_hit_rate"] == 1.0
    assert abs(metrics["prefetch_coverage"] - 1 / 3) < 1e-9


def test_prefetch_bytes_are_not_double_counted_or_mixed_with_placement_bytes():
    trace = ResidencyTrace()
    trace.record(_ev(1.0, "a", "prefetch", bytes_moved=0))
    trace.record(_ev(2.0, "a", "prefetch-complete", bytes_moved=100))
    metrics = trace.metrics()
    assert metrics["bytes_prefetched"] == 100
    assert metrics["bytes_moved"] == 0


def test_skipped_and_error_prefetches_are_counted_separately():
    trace = ResidencyTrace()
    trace.record(_ev(1.0, "a", "prefetch"))
    trace.record(_ev(1.1, "a", "prefetch-skipped"))
    trace.record(_ev(2.0, "b", "prefetch"))
    trace.record(_ev(2.1, "b", "prefetch-error"))
    metrics = trace.metrics()
    assert (metrics["prefetch_skipped"], metrics["prefetch_errors"], metrics["prefetch_completed"]) == (1, 1, 0)


def test_jsonl_persistence_keeps_one_handle_and_close_flushes(tmp_path):
    path = tmp_path / "nested" / "trace.jsonl"
    trace = ResidencyTrace(jsonl_path=str(path))
    trace.record(_ev(1.0, "a", "prefetch"))
    trace.record(_ev(2.0, "a", "resident", bytes_moved=5, dst=MemoryTier.VRAM))
    handle = trace._handle
    trace.record(_ev(3.0, "a", "evict"))
    assert trace._handle is handle
    trace.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["action"] for line in lines] == ["prefetch", "resident", "evict"]
    assert lines[1]["destination"] == "vram"


def test_model_registry_resolves_environment_override(tmp_path):
    spec = get_local_model("3B")
    assert spec.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert spec.resolve_path({"NSA_QWEN_3B_PATH": "/models/qwen3b"}) == "/models/qwen3b"
    root = tmp_path / "cache" / "snapshots"
    snapshot = root / "abc123"
    snapshot.mkdir(parents=True)
    assert spec.__class__(spec.key, spec.model_id, spec.env_var, str(root.parent), spec.vram_budget_gb, spec.ram_budget_gb).checkpoint_path() == str(snapshot)


def _spec(tmp_path):
    spec = get_local_model("1.5b")
    return spec.__class__(spec.key, spec.model_id, spec.env_var, str(tmp_path / "models--x"), 1.0, 1.0)


def test_registry_prefers_refs_main_over_lexicographic_order(tmp_path):
    spec = _spec(tmp_path)
    root = tmp_path / "models--x"
    for name in ("aaa", "zzz"):
        (root / "snapshots" / name).mkdir(parents=True)
        (root / "snapshots" / name / "config.json").write_text("{}")
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("aaa\n")
    assert spec.checkpoint_path().endswith("snapshots/aaa")
    assert spec.is_local()


def test_registry_skips_incomplete_snapshots_and_reports_not_local(tmp_path):
    spec = _spec(tmp_path)
    root = tmp_path / "models--x" / "snapshots"
    (root / "complete").mkdir(parents=True)
    (root / "complete" / "config.json").write_text("{}")
    (root / "partial").mkdir()
    assert spec.checkpoint_path().endswith("snapshots/complete")
    (root / "complete" / "config.json").unlink()
    assert not spec.is_local()


def test_registry_honours_hf_cache_environment(tmp_path):
    spec = get_local_model("3b")
    assert spec.resolve_path({"HF_HOME": str(tmp_path)}) == str(tmp_path / "hub" / "models--Qwen--Qwen2.5-3B-Instruct")
    assert spec.resolve_path({"HF_HUB_CACHE": str(tmp_path / "c")}) == str(tmp_path / "c" / "models--Qwen--Qwen2.5-3B-Instruct")
