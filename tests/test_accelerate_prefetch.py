import os
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
    # freshly written files are already page-cache resident; skip_if_resident=False
    # isolates this test to format parsing rather than the residency-skip behaviour
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES, skip_if_resident=False)
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
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES, skip_if_resident=False)
    assert prefetcher.prefetch("layer.1") == 64
    assert prefetcher.prefetch("layer.2") == 32


def test_huggingface_weight_map_and_single_file_layouts(tmp_path):
    shard = tmp_path / "shard-1.safetensors"
    _write_safetensors(shard, {"model.layers.1.w": b"z" * 20})
    (tmp_path / "index.json").write_text(json.dumps({"weight_map": {"model.layers.1.w": shard.name}}))
    assert AccelerateDiskPrefetcher(tmp_path, PREFIXES, skip_if_resident=False).prefetch("layer.1") == 20

    single = tmp_path / "single"
    single.mkdir()
    _write_safetensors(single / "model.safetensors", {"model.layers.2.w": b"q" * 8})
    assert AccelerateDiskPrefetcher(single, PREFIXES, skip_if_resident=False).prefetch("layer.2") == 8


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
    # offload_weight() just wrote these files, so they're already page-cache
    # resident; skip_if_resident=False isolates this test to format parsing.
    prefetcher = AccelerateDiskPrefetcher(tmp_path, PREFIXES, skip_if_resident=False)
    assert prefetcher.available
    assert prefetcher.prefetch("layer.1") == 64 * 64 * 2


class TestSkipsAlreadyResidentBytes:
    """Regression: the prefetcher used to re-read every mapped byte on every
    call regardless of whether it was already in the OS page cache, which
    measured as pure overhead on a real model (see docs/ACTIVE_RESIDENCY.md)."""

    @staticmethod
    def _evict(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # length 0 means "to EOF"
        finally:
            os.close(fd)

    @pytest.fixture(autouse=True)
    def _skip_without_fadvise(self):
        if not hasattr(os, "posix_fadvise"):
            pytest.skip("posix_fadvise is POSIX-only")

    def test_cold_region_is_read_warm_region_is_not(self, tmp_path):
        key = "model.layers.1.w"
        data = os.urandom(2 * 1024 * 1024)
        (tmp_path / f"{key}.dat").write_bytes(data)
        (tmp_path / "index.json").write_text(json.dumps({key: {"dtype": "float16", "shape": [len(data)]}}))
        self._evict(str(tmp_path / f"{key}.dat"))

        prefetcher = AccelerateDiskPrefetcher(tmp_path, {"layer.1": ("model.layers.1.",)})
        first = prefetcher.prefetch("layer.1")
        assert first == len(data)  # cold: actually read
        assert prefetcher.bytes_already_resident == 0

        second = prefetcher.prefetch("layer.1")
        assert second == 0  # warm from the first call: nothing new read
        assert prefetcher.bytes_already_resident == len(data)
        assert prefetcher.bytes_prefetched == len(data)  # unchanged: no new bytes read

    def test_skip_if_resident_false_always_rereads(self, tmp_path):
        key = "model.layers.1.w"
        data = os.urandom(1024 * 1024)
        (tmp_path / f"{key}.dat").write_bytes(data)
        (tmp_path / "index.json").write_text(json.dumps({key: {"dtype": "float16", "shape": [len(data)]}}))
        prefetcher = AccelerateDiskPrefetcher(tmp_path, {"layer.1": ("model.layers.1.",)}, skip_if_resident=False)
        assert prefetcher.prefetch("layer.1") == len(data)
        assert prefetcher.prefetch("layer.1") == len(data)  # re-read every time, as before this change
        assert prefetcher.bytes_already_resident == 0

    def test_partially_resident_region_only_rereads_what_it_must(self, tmp_path):
        """Two separate tensors backing one region: warm one, evict the other."""
        warm_key, cold_key = "model.layers.2.a", "model.layers.2.b"
        warm_data, cold_data = os.urandom(1024 * 1024), os.urandom(1024 * 1024)
        (tmp_path / f"{warm_key}.dat").write_bytes(warm_data)
        (tmp_path / f"{cold_key}.dat").write_bytes(cold_data)
        (tmp_path / "index.json").write_text(json.dumps({
            warm_key: {"dtype": "float16", "shape": [len(warm_data)]},
            cold_key: {"dtype": "float16", "shape": [len(cold_data)]},
        }))
        self._evict(str(tmp_path / f"{warm_key}.dat"))
        self._evict(str(tmp_path / f"{cold_key}.dat"))
        prefetcher = AccelerateDiskPrefetcher(tmp_path, {"layer.2": ("model.layers.2.",)})
        with open(tmp_path / f"{warm_key}.dat", "rb") as handle:
            handle.read()  # warm only one of the two files backing this region

        touched = prefetcher.prefetch("layer.2")
        assert touched == len(cold_data)  # only the cold file was actually read
        assert prefetcher.bytes_already_resident == len(warm_data)
