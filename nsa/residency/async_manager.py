"""Asynchronous, backend-neutral neural residency manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping

from .residency_policy import NextUseEvictionPolicy, ResidentEntry


@dataclass
class ResidencyMetrics:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    prefetches: int = 0
    cancelled_prefetches: int = 0
    evictions: int = 0
    bytes_loaded: int = 0
    load_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass
class _Resident:
    value: Any
    size_bytes: int
    priority: float = 0.0
    next_use: int | None = None


Loader = Callable[[str], Awaitable[Any] | Any]
Sizer = Callable[[str], int]


class AsyncResidencyManager:
    """Coordinate loading, caching, prefetching and eviction.

    The manager owns policy and lifecycle only. A loader may use filesystem,
    NVMe, DMA, GPU APIs, NPU APIs, or another transport without changing this
    component.
    """

    def __init__(
        self,
        loader: Loader,
        size_of: Sizer,
        capacity_bytes: int,
        max_concurrent_loads: int = 2,
        policy: NextUseEvictionPolicy | None = None,
    ):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if max_concurrent_loads <= 0:
            raise ValueError("max_concurrent_loads must be positive")
        self.loader = loader
        self.size_of = size_of
        self.capacity_bytes = capacity_bytes
        self.policy = policy or NextUseEvictionPolicy()
        self.metrics = ResidencyMetrics()
        self._resident: dict[str, _Resident] = {}
        self._inflight: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()
        self._load_slots = asyncio.Semaphore(max_concurrent_loads)

    @property
    def resident_bytes(self) -> int:
        return sum(item.size_bytes for item in self._resident.values())

    def _entries(self) -> Mapping[str, ResidentEntry]:
        return {
            name: ResidentEntry(
                name,
                item.size_bytes,
                item.next_use,
                item.priority,
            )
            for name, item in self._resident.items()
        }

    async def _load(self, region: str) -> Any:
        async with self._load_slots:
            started = perf_counter()
            result = self.loader(region)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                result = await result
            self.metrics.load_seconds += perf_counter() - started
            self.metrics.bytes_loaded += self.size_of(region)
            return result

    async def _ensure_capacity(self, required: int, protected: set[str]) -> None:
        while self.resident_bytes + required > self.capacity_bytes:
            candidates = {
                name: entry
                for name, entry in self._entries().items()
                if name not in protected
            }
            victim = self.policy.choose_victim(
                candidates, required, self.capacity_bytes
            )
            if victim is None:
                raise MemoryError(
                    f"cannot admit region requiring {required} bytes into "
                    f"{self.capacity_bytes}-byte residency budget"
                )
            self._resident.pop(victim, None)
            self.metrics.evictions += 1

    async def get(
        self,
        region: str,
        *,
        priority: float = 0.0,
        next_use: int | None = None,
    ) -> Any:
        async with self._lock:
            self.metrics.requests += 1
            cached = self._resident.get(region)
            if cached is not None:
                cached.priority = priority
                cached.next_use = next_use
                self.metrics.hits += 1
                return cached.value
            self.metrics.misses += 1

            task = self._inflight.get(region)
            if task is None:
                task = asyncio.create_task(self._load(region))
                self._inflight[region] = task

        try:
            value = await task
        except asyncio.CancelledError:
            raise
        finally:
            async with self._lock:
                if self._inflight.get(region) is task:
                    self._inflight.pop(region, None)

        size = self.size_of(region)
        async with self._lock:
            await self._ensure_capacity(size, {region})
            self._resident[region] = _Resident(value, size, priority, next_use)
        return value

    async def prefetch(self, regions: list[str]) -> None:
        for region in regions:
            async with self._lock:
                if region in self._resident or region in self._inflight:
                    continue
                task = asyncio.create_task(self._load(region))
                self._inflight[region] = task
                self.metrics.prefetches += 1

            async def finish(name: str, pending: asyncio.Task[Any]) -> None:
                try:
                    value = await pending
                    size = self.size_of(name)
                    async with self._lock:
                        await self._ensure_capacity(size, set())
                        self._resident[name] = _Resident(value, size)
                except (asyncio.CancelledError, MemoryError):
                    if pending.cancelled():
                        self.metrics.cancelled_prefetches += 1
                finally:
                    async with self._lock:
                        if self._inflight.get(name) is pending:
                            self._inflight.pop(name, None)

            asyncio.create_task(finish(region, task))

    async def evict(self, region: str) -> bool:
        async with self._lock:
            return self._resident.pop(region, None) is not None

    async def clear(self) -> None:
        async with self._lock:
            for task in self._inflight.values():
                task.cancel()
            self._inflight.clear()
            self._resident.clear()
