import asyncio
from pathlib import Path

from nsa.residency.range_store import LocalRangeSource, MmapRangeSource, TensorRangeStore
from nsa.residency.weight_index import TensorRegion, WeightIndex


def test_local_range_store_reads_only_tensor_payload(tmp_path):
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"HEADER" + b"AAAA" + b"BBBBBB")
    index = WeightIndex((
        TensorRegion("a", path.name, 4, (4,), "U8", "layer.0", 6),
        TensorRegion("b", path.name, 6, (6,), "U8", "layer.0", 10),
    ), {path.name: path.stat().st_size})
    store = TensorRangeStore(index, LocalRangeSource())

    loaded = asyncio.run(store.load_region_bytes("layer.0"))

    assert loaded == {"a": b"AAAA", "b": b"BBBBBB"}


def test_mmap_range_store_reads_exact_range(tmp_path):
    path = tmp_path / "shard.safetensors"
    path.write_bytes(b"0123456789")
    index = WeightIndex((
        TensorRegion("a", path.name, 3, (3,), "U8", "layer.0", 4),
    ), {path.name: 10})
    store = TensorRangeStore(index, MmapRangeSource())

    assert asyncio.run(store.load_tensor_bytes(index.tensors[0])) == b"456"
