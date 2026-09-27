"""Metadata-first safetensors inspection and region sizing."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .weight_index import WeightIndex, from_tensor_metadata


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = handle.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"truncated safetensors header in {path}")
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid safetensors header object in {path}")
    return data


def inspect_safetensors_metadata(root: str | Path, index: WeightIndex) -> WeightIndex:
    """Read safetensors headers without materializing tensor payloads."""
    root_path = Path(root)
    indexed = {(t.name, t.shard) for t in index.tensors}
    metadata: dict[str, dict[str, Any]] = {}
    shard_sizes = dict(index.shard_sizes)

    for shard in dict.fromkeys(t.shard for t in index.tensors):
        path = root_path / shard
        shard_sizes.setdefault(shard, path.stat().st_size)
        header = _read_header(path)
        for name, entry in header.items():
            if name == "__metadata__" or (name, shard) not in indexed:
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"invalid tensor metadata for {name!r}")
            shape = tuple(int(v) for v in entry.get("shape", ()))
            dtype = str(entry.get("dtype", "unknown")).upper()
            metadata[name] = {"shape": shape, "dtype": dtype}

    return from_tensor_metadata(
        {"weight_map": {t.name: t.shard for t in index.tensors}},
        metadata,
        shard_sizes,
    )
