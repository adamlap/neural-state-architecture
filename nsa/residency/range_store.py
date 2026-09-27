"""Tensor-granular local and HTTP-range checkpoint access."""

from __future__ import annotations

import asyncio
import mmap
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from .weight_index import TensorRegion, WeightIndex


class ByteRangeSource(Protocol):
    async def read(self, shard: str, start: int, end: int) -> bytes:
        ...


class LocalRangeSource:
    """Read only the requested payload range from a local shard."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    async def read(self, shard: str, start: int, end: int) -> bytes:
        return await asyncio.to_thread(self._read, self.root / shard, start, end)

    @staticmethod
    def _read(path: str | Path, start: int, end: int) -> bytes:
        with open(path, "rb") as handle:
            handle.seek(start)
            data = handle.read(end - start)
        if len(data) != end - start:
            raise IOError(f"short read: expected {end - start}, got {len(data)}")
        return data


class MmapRangeSource:
    """Zero-copy-oriented local range source backed by memory mapping."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)

    async def read(self, shard: str, start: int, end: int) -> bytes:
        return await asyncio.to_thread(self._read, self.root / shard, start, end)

    @staticmethod
    def _read(path: str | Path, start: int, end: int) -> bytes:
        length = end - start
        with open(path, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                return mapped[start:start + length]


class HttpRangeSource:
    """HTTP Range source for remote checkpoints."""

    def __init__(self, urls: dict[str, str], timeout: float = 30.0):
        self.urls = urls
        self.timeout = timeout

    async def read(self, shard: str, start: int, end: int) -> bytes:
        return await asyncio.to_thread(self._read, shard, start, end)

    def _read(self, shard: str, start: int, end: int) -> bytes:
        request = Request(
            self.urls[shard],
            headers={"Range": f"bytes={start}-{end - 1}"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            data = response.read()
        if len(data) != end - start:
            raise IOError(
                f"short HTTP range: expected {end - start}, got {len(data)}"
            )
        return data


class TensorRangeStore:
    """Load exactly the bytes belonging to selected indexed tensors."""

    def __init__(self, index: WeightIndex, source: ByteRangeSource):
        self.index = index
        self.source = source

    def tensors_for_region(self, region: str) -> tuple[TensorRegion, ...]:
        return tuple(t for t in self.index.tensors if t.region == region)

    async def load_tensor_bytes(self, tensor: TensorRegion) -> bytes:
        if tensor.data_offset is None:
            raise ValueError(f"tensor {tensor.name!r} has no data offset")
        return await self.source.read(
            tensor.shard,
            tensor.data_offset,
            tensor.data_offset + tensor.parameter_bytes,
        )

    async def load_region_bytes(self, region: str) -> dict[str, bytes]:
        tensors = self.tensors_for_region(region)
        loaded = await asyncio.gather(
            *(self.load_tensor_bytes(tensor) for tensor in tensors)
        )
        return {tensor.name: data for tensor, data in zip(tensors, loaded)}

    async def load_regions_bytes(self, regions: list[str]) -> dict[str, dict[str, bytes]]:
        loaded = await asyncio.gather(*(self.load_region_bytes(r) for r in regions))
        return dict(zip(regions, loaded))
