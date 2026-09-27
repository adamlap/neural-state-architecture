"""Asynchronous, backend-neutral neural residency manager."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

from .residency_policy import NextUseEvictionPolicy, ResidentEntry


@dataclass
class ResidencyMetrics:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    prefetches: int = 0
    cancelled_prefetches: int = 0
    failed_prefetches: int = 0
    evictions: int = 0
    bytes_loaded: int = 0
    load_errors: int = 0
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


Loader = Callable[[str], Any]
Sizer = Callable[[str], int]


class AsyncResidencyManager:
    """Coordinate loading, caching, prefetching and eviction."""

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
        self._background: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()
        self._load_slots = asyncio.Semaphore(max_concurrent_loads)

    @property
    def resident_bytes(self) -> int:
        return sum(item.size_bytes for item in self._resident.values())

    def _entries(self) -> Mapping[str, ResidentEntry]:
        return {
            name: ResidentEntry(name, item.size_bytes, item.next_use, item.priority)
            for name, item in self._resident.items()
        }

    async def _load(self, region: str) -> Any:
        async with self._load_slots:
            started = perf_counter()
            try:
                result = self.loader(region)
                if inspect.isawaitable(result):
                    result = await result
                self.metrics.bytes_loaded += self.size_of(region)
                return result
            except Exception:
                self.metrics.load_errors += 1
                raise
            finally:
                self.metrics.load_seconds += perf_counter() - started

    async def _ensure_capacity(self, required: int, protected: set[str]) -> None:
        if required > self.capacity_bytes:
            raise MemoryError(
                f"region requires {required} bytes but residency budget is "
                f"{self.capacity_bytes} bytes"
            )
        while self.resident_bytes + required > self.capacity_bytes:
            candidates = {
                name: entry
                for name, entry in self._entries().items()
                if name not in protected
            }
            victim = self.policy.choose_victim(candidates, required, self.capacity_bytes)
            if victim is None:
                raise MemoryError(
                    f"cannot admit region requiring {required} bytes into "
                    f"{self.capacity_bytes}-byte residency budget"
                )
            self._resident.pop(victim, None)
            self.metrics.evictions += 1

    async def get(self, region: str, *, priority: float = 0.0, next_use: int | None = None) -> Any:
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
        finally:
            async with self._lock:
                if self._inflight.get(region) is task:
                    self._inflight.pop(region, None)

        size = self.size_of(region)
        async with self._lock:
            await self._ensure_capacity(size, {region})
            self._resident[region] = _Resident(value, size, priority, next_use)
        return value

    async def _finish_prefetch(self, name: str, pending: asyncio.Task[Any]) -> None:
        try:
            value = await pending
            size = self.size_of(name)
            async with self._lock:
                await self._ensure_capacity(size, set())
                self._resident[name] = _Resident(value, size)
        except asyncio.CancelledError:
            self.metrics.cancelled_prefetches += 1
        except Exception:
            self.metrics.failed_prefetches += 1
        finally:
            async with self._lock:
                if self._inflight.get(name) is pending:
                    self._inflight.pop(name, None)

    async def prefetch(self, regions: list[str]) -> None:
        for region in regions:
            async with self._lock:
                if region in self._resident or region in self._inflight:
                    continue
                task = asyncio.create_task(self._load(region))
                self._inflight[region] = task
                self.metrics.prefetches += 1
                background = asyncio.create_task(self._finish_prefetch(region, task))
                self._background.add(background)
                background.add_done_callback(self._background.discard)

    async def drain_prefetches(self) -> None:
        """Wait for currently scheduled prefetches to settle."""
        tasks = tuple(self._background)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def evict(self, region: str) -> bool:
        async with self._lock:
            removed = self._resident.pop(region, None)
            if removed is None:
                return False
            self.metrics.evictions += 1
            return True

    async def clear(self) -> None:
        async with self._lock:
            tasks = tuple(self._inflight.values())
            for task in tasks:
                task.cancel()
            self._inflight.clear()
            self._resident.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        background = tuple(self._background)
        for task in background:
            task.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        self._background.clear()
