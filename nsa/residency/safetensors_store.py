"""Lazy tensor access for hardware-agnostic residency.

Safetensors supports zero-copy/lazy access and multiple backends; this adapter
keeps that capability behind NSA's RegionStore boundary. See Hugging Face's
Safetensors documentation for the underlying format and API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .weight_index import TensorRegion, WeightIndex

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


class SafetensorsRegionStore:
    """Open only the shard needed for a requested tensor region."""

    def __init__(self, root: str | Path, index: WeightIndex):
        self.root = Path(root)
        self.index = index
        self._handles: dict[str, Any] = {}

    def _handle(self, shard: str) -> Any:
        if safe_open is None:
            raise RuntimeError("safetensors is required for SafetensorsRegionStore")
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.root / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def tensors_for_region(self, region: str) -> tuple[TensorRegion, ...]:
        return tuple(t for t in self.index.tensors if t.region == region)

    def load_region(self, region: str) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for tensor in self.tensors_for_region(region):
            loaded[tensor.name] = self._handle(tensor.shard).get_tensor(tensor.name)
        return loaded

    def close(self) -> None:
        for handle in self._handles.values():
            close = getattr(handle, "close", None)
            if close is not None:
                close()
        self._handles.clear()
