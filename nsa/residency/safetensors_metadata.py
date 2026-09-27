"""Metadata-first safetensors inspection and region sizing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .weight_index import WeightIndex, from_tensor_metadata

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


def inspect_safetensors_metadata(root: str | Path, index: WeightIndex) -> WeightIndex:
    """Read tensor headers without materializing tensor payloads.

    safe_open exposes shape/dtype metadata directly. This lets NSA compute
    exact region sizes before admitting anything into RAM/VRAM.
    """
    if safe_open is None:
        raise RuntimeError("safetensors is required for metadata inspection")

    metadata: dict[str, dict[str, Any]] = {}
    shard_sizes = dict(index.shard_sizes)
    indexed = {(t.name, t.shard) for t in index.tensors}
    for shard in dict.fromkeys(t.shard for t in index.tensors):
        path = Path(root) / shard
        shard_sizes.setdefault(shard, path.stat().st_size)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if (name, shard) not in indexed:
                    continue
                tensor = handle.get_slice(name)
                metadata[name] = {
                    "shape": tuple(int(v) for v in tensor.get_shape()),
                    "dtype": str(handle.get_dtype(name)).upper(),
                }
    return from_tensor_metadata(
        {"weight_map": {t.name: t.shard for t in index.tensors}},
        metadata,
        shard_sizes,
    )
