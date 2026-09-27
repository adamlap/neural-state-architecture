"""Metadata-first safetensors inspection and region sizing."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .weight_index import WeightIndex, from_tensor_metadata


_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _read_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError(f"safetensors header too large in {path}")
        header = handle.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"truncated safetensors header in {path}")
    data = json.loads(header.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid safetensors header object in {path}")
    return data, header_length


def inspect_safetensors_metadata(root: str | Path, index: WeightIndex) -> WeightIndex:
    """Read safetensors headers without materializing tensor payloads.

    The data offsets are authoritative for tensor payload size, so this stays
    exact for packed and quantized formats with non-standard dtype strings.
    """
    root_path = Path(root)
    indexed = {(t.name, t.shard) for t in index.tensors}
    metadata: dict[str, dict[str, Any]] = {}
    shard_sizes = dict(index.shard_sizes)

    for shard in dict.fromkeys(t.shard for t in index.tensors):
        path = root_path / shard
        shard_size = path.stat().st_size
        header, header_length = _read_header(path)
        data_start = 8 + header_length
        data_end = shard_size - data_start
        if data_end < 0:
            raise ValueError(f"invalid safetensors data section in {path}")
        shard_sizes.setdefault(shard, shard_size)

        for name, entry in header.items():
            if name == "__metadata__" or (name, shard) not in indexed:
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"invalid tensor metadata for {name!r}")
            offsets = entry.get("data_offsets")
            if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
                raise ValueError(f"missing data_offsets for {name!r}")
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or end > data_end:
                raise ValueError(f"invalid data_offsets for {name!r}")
            shape = tuple(int(v) for v in entry.get("shape", ()))
            dtype = str(entry.get("dtype", "unknown")).upper()
            metadata[name] = {
                "shape": shape,
                "dtype": dtype,
                "data_offsets": (start, end),
            }

    return from_tensor_metadata(
        {"weight_map": {t.name: t.shard for t in index.tensors}},
        metadata,
        shard_sizes,
    )
