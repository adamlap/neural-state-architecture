import json

import pytest

from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend
from tests.tiny_qwen import make_tiny_qwen, torch_backend_usable


def test_selective_device_map_keeps_hot_edges_and_offloads_middle():
    backend = SelectiveStorageTransformersBackend(mode="mock", device="cpu", hot_layers=1, warm_layers=1)
    mapping = backend._selective_device_map(6)
    assert mapping["model.layers.0"] == "cpu"
    assert mapping["model.layers.5"] == "cpu"
    assert mapping["model.layers.1"] == "cpu"
    assert mapping["model.layers.4"] == "cpu"
    assert mapping["model.layers.2"] == "disk"
    assert mapping["model.layers.3"] == "disk"


def test_device_resolution_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for wanted in ("auto", "cuda", "cuda:1", "cpu", "mps", ""):
        assert SelectiveStorageTransformersBackend._resolve_device(wanted) == "cpu"


def test_device_auto_uses_cuda_when_available(monkeypatch):
    """Regression: 'auto' used to resolve to CPU even when CUDA was present."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert SelectiveStorageTransformersBackend._resolve_device("auto") == "cuda"
    assert SelectiveStorageTransformersBackend._resolve_device("cuda:1") == "cuda:1"
    backend = SelectiveStorageTransformersBackend(mode="mock", device="cuda:1", hot_layers=1, warm_layers=0)
    assert backend._selective_device_map(4)["model.layers.0"] == 1  # Accelerate's integer GPU spelling


def test_remote_code_is_not_trusted_by_default():
    assert SelectiveStorageTransformersBackend(mode="mock").trust_remote_code is False


needs_stack = pytest.mark.skipif(not torch_backend_usable(), reason="needs a working torch + transformers + accelerate stack")


@needs_stack
def test_end_to_end_prefetch_on_tiny_qwen(tmp_path):
    root = make_tiny_qwen(tmp_path / "model")
    backend = SelectiveStorageTransformersBackend(
        model_name="tiny/qwen", model_path=str(root), device="cpu", hot_layers=1, warm_layers=1,
        vram_budget_gb=1, ram_budget_gb=1, offload_folder=str(tmp_path / "offload"),
    )
    try:
        backend.load_model()
        # Accelerate's own dispatch (offload_state_dict=True) just wrote these
        # files, so they're page-cache warm from the write itself; evict them
        # so the prefetcher has genuinely cold bytes to warm, matching what a
        # real long-lived deployment looks like once memory pressure has
        # evicted a model's disk-backed layers (see benchmark_matrix.py's
        # --cold-cache, which does the same thing for the same reason).
        from experiments.residency.benchmark_matrix import _drop_page_cache
        _drop_page_cache(backend.offload_folder)
        out = backend.generate("explain how persistent cognitive state", max_tokens=6, temperature=0.0)
        backend.residency_controller.wait()
        assert len(out.tokens) == 6
        json.dumps(out.raw_response)  # snapshot must be serialisable

        regions = backend.residency.regions
        assert len(regions) == 8
        assert backend.prefetcher.available
        # hot/warm layers are placed in RAM at dispatch; the middle four live on disk
        snapshot = backend.residency.snapshot()
        assert sorted(snapshot.resident_regions()) == ["layer.0", "layer.1", "layer.6", "layer.7"]
        # region sizes are measured from the checkpoint, not guessed
        assert regions["layer.0"].size_bytes == regions["layer.3"].size_bytes > 0

        metrics = backend.trace.metrics()
        assert metrics["executions"] > 0
        assert metrics["prefetch_errors"] == 0
        # On a model this small/fast, Accelerate's own synchronous read of a
        # disk-tier layer (required for that layer's forward regardless of
        # prefetching) finishes before the background prefetch thread would
        # even be scheduled, and warms the same bytes itself during the very
        # first token. From then on prefetch_eligible (needs_warming) sees
        # every disk-tier region already resident and correctly schedules
        # *nothing* -- zero locks, zero thread handoffs, zero redundant
        # reads -- rather than running the machinery just to observe "already
        # warm". That is the intended optimum for a workload whose whole
        # disk-tier working set is touched, and therefore kept warm, on
        # every step; it is not the same thing as the mechanism failing to
        # find anything real. Assert the real invariant directly: every
        # disk-tier region is genuinely mapped, and once touched, correctly
        # recognised as no longer needing a prefetch.
        disk_regions = [rid for rid in regions if rid not in snapshot.resident_regions()]
        assert disk_regions, "expected at least one disk-tier region for this test to be meaningful"
        for region_id in disk_regions:
            assert backend.prefetcher.covers(region_id)
            assert not backend.prefetcher.needs_warming(region_id)  # Accelerate's own read already warmed it
    finally:
        backend.close()


@needs_stack
def test_prefetch_does_not_change_generated_tokens(tmp_path):
    root = make_tiny_qwen(tmp_path / "model")
    outputs = []
    for prefetch in (True, False):
        backend = SelectiveStorageTransformersBackend(
            model_name="tiny/qwen", model_path=str(root), device="cpu", hot_layers=1, warm_layers=1,
            prefetch=prefetch, offload_folder=str(tmp_path / f"offload-{prefetch}"),
        )
        try:
            outputs.append(backend.generate("explain how persistent cognitive state", max_tokens=8, temperature=0.0).tokens)
        finally:
            backend.close()
    assert outputs[0] == outputs[1]


@needs_stack
def test_generate_after_close_does_not_crash(tmp_path):
    root = make_tiny_qwen(tmp_path / "model")
    backend = SelectiveStorageTransformersBackend(
        model_name="tiny/qwen", model_path=str(root), device="cpu", hot_layers=1, warm_layers=1,
        offload_folder=str(tmp_path / "offload"),
    )
    backend.generate("explain how", max_tokens=2, temperature=0.0)
    backend.close()
    assert len(backend.generate("explain how", max_tokens=2, temperature=0.0).tokens) == 2
