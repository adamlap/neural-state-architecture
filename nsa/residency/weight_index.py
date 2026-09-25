"""Checkpoint-index inspection without loading model weights.

The planner understands the public Safetensors index format and converts tensor
metadata into logical residency records. It intentionally performs no I/O
against the tensor payloads, making it suitable for very large checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorRegion:
    name: str
    shard: str
    parameter_bytes: int
    shape: tuple[int, ...]
    dtype: str
    region: str


@dataclass(frozen=True)
class WeightIndex:
    tensors: tuple[TensorRegion, ...]
    shard_sizes: dict[str, int]

    @property
    def total_bytes(self) -> int:
        return sum(t.parameter_bytes for t in self.tensors)

    def regions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.region for t in self.tensors))


def _dtype_size(dtype: str) -> int:
    sizes = {
        "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    }
    if dtype not in sizes:
        raise ValueError(f"unsupported or unknown dtype {dtype!r}")
    return sizes[dtype]


def _tensor_region(name: str) -> str:
    if name.startswith("model.embed_tokens."):
        return "embeddings"
    if ".layers." in name:
        tail = name.split(".layers.", 1)[1]
        index = tail.split(".", 1)[0]
        return f"layer.{index}"
    if name.startswith("model.norm.") or name.startswith("model.rotary_emb."):
        return "model_state"
    if name.startswith("lm_head."):
        return "lm_head"
    if name.startswith("visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def load_safetensors_index(path: str | Path) -> WeightIndex:
    """Parse model.safetensors.index.json without opening any weight shard."""

    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("missing weight_map in safetensors index")

    tensors: list[TensorRegion] = []
    for name, shard in weight_map.items():
        # The index records dtype/shape in some exporters, but not all.  When
        # absent we retain a zero-byte placeholder rather than guessing.
        metadata = data.get("metadata", {})
        _ = metadata
        tensors.append(TensorRegion(
            name=name,
            shard=str(shard),
            parameter_bytes=0,
            shape=(),
            dtype="unknown",
            region=_tensor_region(name),
        ))

    return WeightIndex(tuple(tensors), {})


def from_tensor_metadata(
    index_data: dict[str, Any],
    tensor_metadata: dict[str, dict[str, Any]],
    shard_sizes: dict[str, int] | None = None,
) -> WeightIndex:
    """Build a complete index when tensor header metadata is available."""

    weight_map = index_data.get("weight_map", {})
    tensors: list[TensorRegion] = []
    for name, shard in weight_map.items():
        meta = tensor_metadata.get(name, {})
        dtype = str(meta.get("dtype", "unknown"))
        shape = tuple(int(v) for v in meta.get("shape", ()))
        if dtype == "unknown":
            size = int(meta.get("parameter_bytes", 0))
        else:
            size = 1
            for dimension in shape:
                size *= dimension
            size *= _dtype_size(dtype)
        tensors.append(TensorRegion(
            name=name,
            shard=str(shard),
            parameter_bytes=size,
            shape=shape,
            dtype=dtype,
            region=_tensor_region(name),
        ))
    return WeightIndex(tuple(tensors), dict(shard_sizes or {}))
