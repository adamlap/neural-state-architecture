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
        assert metrics["prefetch_completed"] > 0 and metrics["prefetch_skipped"] == 0
        assert metrics["prefetch_errors"] == 0
        assert metrics["prefetch_hit_rate"] > 0.9  # layer i+1 is warmed while layer i runs
        assert metrics["bytes_prefetched"] == backend.prefetcher.bytes_prefetched > 0
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
