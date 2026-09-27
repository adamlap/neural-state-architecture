"""Tier-aware region loading built on the universal async residency manager."""

from __future__ import annotations

from typing import Any, Protocol

from .async_manager import AsyncResidencyManager, ResidencyMetrics
from .safetensors_store import SafetensorsRegionStore


class RegionSource(Protocol):
    def load_region(self, region: str) -> Any:
        ...
    def size_of(self, region: str) -> int:
        ...


class WeightIndexSource:
    def __init__(self, store: SafetensorsRegionStore):
        self.store = store

    def load_region(self, region: str) -> Any:
        return self.store.load_region(region)

    def size_of(self, region: str) -> int:
        return sum(t.parameter_bytes for t in self.store.tensors_for_region(region))


class TieredRegionStore:
    """Durable source + bounded hot residency."""

    def __init__(
        self,
        source: RegionSource,
        capacity_bytes: int,
        *,
        max_concurrent_loads: int = 2,
    ):
        self.source = source
        self.residency = AsyncResidencyManager(
            source.load_region,
            source.size_of,
            capacity_bytes,
            max_concurrent_loads=max_concurrent_loads,
        )

    @property
    def metrics(self) -> ResidencyMetrics:
        return self.residency.metrics

    @property
    def resident_bytes(self) -> int:
        return self.residency.resident_bytes

    async def get(self, region: str, **kwargs: Any) -> Any:
        return await self.residency.get(region, **kwargs)

    async def prefetch(self, regions: list[str]) -> None:
        await self.residency.prefetch(regions)

    async def drain_prefetches(self) -> None:
        await self.residency.drain_prefetches()

    async def evict(self, region: str) -> bool:
        return await self.residency.evict(region)

    async def clear(self) -> None:
        await self.residency.clear()
