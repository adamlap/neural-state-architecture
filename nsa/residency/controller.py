"""Predictive active residency orchestration."""
from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import monotonic
from typing import Callable, Mapping, Optional, Sequence
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyDecision
from nsa.residency.types import MemoryTier, ResidencyEvent

LoadFn = Callable[[str, MemoryTier], None]
EvictFn = Callable[[str, MemoryTier], None]

@dataclass(frozen=True)
class PrefetchTask:
    region_id: str
    tier: MemoryTier
    score: float
    reason: str

class ActiveResidencyController:
    """Turn NSA residency plans into backend load/evict operations."""
    def __init__(self, manager: NeuralResidencyManager, load_fn: LoadFn, evict_fn: Optional[EvictFn] = None, lookahead: int = 2) -> None:
        self.manager = manager
        self.load_fn = load_fn
        self.evict_fn = evict_fn
        self.lookahead = max(0, lookahead)
        self._lock = Lock()
        self._inflight: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=max(1, lookahead), thread_name_prefix="nsa-residency")
        self._futures: dict[str, Future[None]] = {}

    def _tasks(self, decisions: Sequence[ResidencyDecision]) -> list[PrefetchTask]:
        candidates = [d for d in decisions if d.prefetch and d.desired_tier in (MemoryTier.VRAM, MemoryTier.RAM)]
        return [PrefetchTask(d.region_id, d.desired_tier, d.score, d.reason) for d in candidates[:self.lookahead]]

    def _load(self, task: PrefetchTask) -> None:
        started = monotonic()
        self.manager.record_event(ResidencyEvent(monotonic(), task.region_id, "prefetch", MemoryTier.NVME, task.tier, self.manager.regions[task.region_id].size_bytes, 0.0, "predictive-prefetch-start"))
        self.load_fn(task.region_id, task.tier)
        self.manager.record_resident(task.region_id, task.tier, reason="predictive-prefetch")
        self.manager.record_event(ResidencyEvent(monotonic(), task.region_id, "prefetch-complete", MemoryTier.NVME, task.tier, self.manager.regions[task.region_id].size_bytes, (monotonic()-started)*1000, "predictive-prefetch"))

    def prefetch_async(self, state: Mapping[str, object]) -> list[PrefetchTask]:
        """Schedule predictive loads without blocking the inference thread."""
        tasks = self._tasks(self.manager.plan(state))
        for task in tasks:
            with self._lock:
                if task.region_id in self._inflight:
                    continue
                self._inflight.add(task.region_id)
                future = self._executor.submit(self._load, task)
                self._futures[task.region_id] = future
                future.add_done_callback(lambda _, rid=task.region_id: self._finish(rid))
        return tasks

    def _finish(self, region_id: str) -> None:
        with self._lock:
            self._inflight.discard(region_id)
            self._futures.pop(region_id, None)

    def wait(self, region_id: Optional[str] = None, timeout: Optional[float] = None) -> None:
        """Wait for one region or all currently scheduled prefetches."""
        futures = [self._futures.get(region_id)] if region_id else list(self._futures.values())
        for future in futures:
            if future is not None:
                future.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def tick(self, state: Mapping[str, object]) -> list[PrefetchTask]:
        decisions = self.manager.plan(state)
        tasks = self._tasks(decisions)
        started = monotonic()
        for task in tasks:
            with self._lock:
                if task.region_id in self._inflight:
                    continue
                self._inflight.add(task.region_id)
            try:
                self._load(task)
            finally:
                with self._lock:
                    self._inflight.discard(task.region_id)
        self.manager.record_event(ResidencyEvent(monotonic(), "*", "prefetch-cycle", None, None, 0, (monotonic()-started)*1000, f"tasks={len(tasks)}"))
        return tasks

    def retain_or_evict(self, decisions: Sequence[ResidencyDecision]) -> list[str]:
        retained = []
        desired = {d.region_id: d for d in decisions}
        for region_id, tier in list(self.manager.tiers.items()):
            if self.manager.states.get(region_id).value != "resident":
                continue
            decision = desired.get(region_id)
            if decision and decision.retain:
                retained.append(region_id)
                continue
            if self.evict_fn and tier in (MemoryTier.VRAM, MemoryTier.RAM):
                self.evict_fn(region_id, tier)
            self.manager.record_evicted(region_id, reason="policy")
        return retained
