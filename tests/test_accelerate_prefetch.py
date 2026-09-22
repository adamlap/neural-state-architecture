import json
import struct

import pytest

from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher, read_safetensors_header

PREFIXES = {"layer.1": ("model.layers.1.",), "layer.2": ("model.layers.2.",)}


def _write_safetensors(path, tensors):
    """Minimal safetensors writer: tensors is {name: bytes}. Keeps tests torch-free."""
    header, blob, cursor = {}, b"", 0
    for name, data in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": [cursor, cursor + len(data)]}
        blob += data
        cursor += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def test_read_safetensors_header_gives_absolute_byte_ranges(tmp_path):
    path = tmp_path / "m.safetensors"
    _write_safetensors(path, {"a": b"x" * 10, "b": b"y" * 6})
    spans = read_safetensors_header(path)
    data = path.read_bytes()
    assert data[spans["a"][0]:spans["a"][1]] == b"x" * 10
    assert data[spans["b"][0]:spans["b"][1]] == b"y" * 6


def test_dat_offload_format_is_warmed_by_region(tmp_path):
    """Accelerate's own format: index entries carry only dtype/shape, weights live in <key>.dat."""
    index = {}
    for layer in (1, 2):
        key = f"model.layers.{layer}.mlp.down_proj.weight"
        (tmp_path / f"{key}.dat").write_bytes(b"\0" * (100 * layer))
        index[key] = {"dtype": "float16", "shape": [10, 5 * layer]}
    (tmp_path / "index.json").write_text(json.dumps(index))
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES)
    assert prefetcher.available
    assert prefetcher.covers("layer.1") and not prefetcher.covers("layer.9")
    assert prefetcher.prefetch("layer.1") == 100
    assert prefetcher.prefetch("layer.2") == 200
    assert prefetcher.bytes_prefetched == 300


def test_safetensors_file_index_entries_read_only_the_tensor_range(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"model.layers.1.w": b"a" * 64, "model.layers.2.w": b"b" * 32})
    index = {
        "model.layers.1.w": {"safetensors_file": str(path), "weight_name": "model.layers.1.w", "dtype": "float16"},
        "model.layers.2.w": {"safetensors_file": str(path), "weight_name": "model.layers.2.w", "dtype": "float16"},
    }
    (tmp_path / "index.json").write_text(json.dumps(index))
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES)
    assert prefetcher.prefetch("layer.1") == 64
    assert prefetcher.prefetch("layer.2") == 32


def test_huggingface_weight_map_and_single_file_layouts(tmp_path):
    shard = tmp_path / "shard-1.safetensors"
    _write_safetensors(shard, {"model.layers.1.w": b"z" * 20})
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {"model.layers.1.w": shard.name}}))
    assert AccelerateDiskPrefetcher(tmp_path, PREFIXES).prefetch("layer.1") == 20

    single = tmp_path / "single"
    single.mkdir()
    _write_safetensors(single / "model.safetensors", {"model.layers.2.w": b"q" * 8})
    assert AccelerateDiskPrefetcher(single, PREFIXES).prefetch("layer.2") == 8


def test_unknown_region_and_missing_files_warm_nothing(tmp_path):
    key = "model.layers.1.w"
    (tmp_path / "index.json").write_text(json.dumps({key: {"dtype": "float16", "shape": [4]}}))
    prefetcher = AccelerateDiskPrefetcher(tmp_path, {"layer.1": ("model.layers.1.",), "empty": ()})
    assert not prefetcher.available  # .dat file is missing
    assert prefetcher.prefetch("layer.1") == 0
    assert prefetcher.prefetch("nope") == 0
    assert prefetcher.prefetch("empty") == 0  # no prefixes must never mean "the whole model"


def test_missing_offload_dir_is_a_safe_noop(tmp_path):
    prefetcher = AccelerateDiskPrefetcher(tmp_path / "absent", PREFIXES)
    assert not prefetcher.available and prefetcher.prefetch("layer.1") == 0


def test_unreadable_extent_is_counted_not_raised(tmp_path):
    key = "model.layers.1.w"
    (tmp_path / f"{key}.dat").write_bytes(b"\0" * 16)
    (tmp_path / "index.json").write_text(json.dumps({key: {"dtype": "float16", "shape": [8]}}))
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES)
    (tmp_path / f"{key}.dat").unlink()
    assert prefetcher.prefetch("layer.1") == 0
    assert prefetcher.read_errors == 1


def test_real_accelerate_offload_output_is_understood(tmp_path):
    """Regression: the prefetcher used to expect 'filename'/'weight_map' and warmed nothing."""
    torch = pytest.importorskip("torch")
    utils = pytest.importorskip("accelerate.utils")
    index = {}
    for layer in (1, 2):
        utils.offload_weight(torch.randn(64, 64).bfloat16(), f"model.layers.{layer}.mlp.down_proj.weight", str(tmp_path), index=index)
    utils.save_offload_index(index, str(tmp_path))
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES)
    assert prefetcher.available
    assert prefetcher.prefetch("layer.1") == 64 * 64 * 2
