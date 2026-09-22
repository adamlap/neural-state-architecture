"""Neural virtual-memory manager."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from time import monotonic
from nsa.residency.cache import CacheEntry, ResidencyCache
from nsa.residency.policy import ResidencyDecision, ResidencyPolicy
from nsa.residency.predictor import HeuristicResidencyPredictor, ResidencyPredictor, state_tags
from nsa.residency.types import MemoryTier, NeuralRegion, ResidencyEvent, ResidencySnapshot, ResidencyState

@dataclass
class NeuralResidencyManager:
    policy: ResidencyPolicy
    predictor: ResidencyPredictor = field(default_factory=HeuristicResidencyPredictor)
    trace: Any = field(default=None, repr=False)
    max_events: int = 10_000
    def __post_init__(self) -> None:
        self.regions: dict[str, NeuralRegion] = {}
        self.states: dict[str, ResidencyState] = {}
        self.tiers: dict[str, MemoryTier] = {}
        self.scores: dict[str, float] = {}
        self.events: deque[ResidencyEvent] = deque(maxlen=max(1, self.max_events))
        self.vram_cache = ResidencyCache(self.policy.vram_budget_bytes)
        self.ram_cache = ResidencyCache(self.policy.ram_budget_bytes)
        # Last region observed *executing*. Residency transfers never change it,
        # otherwise a prefetch would corrupt the execution-transition statistics.
        self.current_region: str | None = None

    def record_event(self, event: ResidencyEvent) -> None:
        self.events.append(event)
        if self.trace is not None:
            self.trace.record(event)
    def register(self, regions: Sequence[NeuralRegion]) -> None:
        for region in regions:
            self.regions[region.region_id] = region
            self.states.setdefault(region.region_id, ResidencyState.COLD)
            self.tiers.setdefault(region.region_id, MemoryTier.NVME)
    def plan(self, state: Mapping[str, object]) -> list[ResidencyDecision]:
        probabilities = self.predictor.predict(tuple(self.regions.values()), state, self.current_region)
        decisions = []
        for region in self.regions.values():
            decision = self.policy.decide(region, self._relevance(region,state), probabilities.get(region.region_id,0.0), self._dependency_probability(region,probabilities))
            self.scores[region.region_id] = decision.score
            decisions.append(decision)
        return sorted(decisions, key=lambda d:d.score, reverse=True)
    def _relevance(self, region: NeuralRegion, state: Mapping[str, object]) -> float | None:
        """Tag overlap in [0, 1], or None when there is no tag evidence either way."""
        tags = {str(t) for t in state_tags(state)}
        if not tags or not region.semantic_tags: return None
        return min(1.0, sum(t in tags for t in region.semantic_tags) / len(region.semantic_tags))
    def _dependency_probability(self, region: NeuralRegion, probabilities: Mapping[str,float]) -> float:
        return max((probabilities.get(dep,0.0) for dep in region.dependencies), default=0.0)
    def _cache_for(self, tier: MemoryTier) -> ResidencyCache | None:
        if tier == MemoryTier.VRAM: return self.vram_cache
        if tier == MemoryTier.RAM: return self.ram_cache
        return None
    def record_resident(self, region_id: str, tier: MemoryTier, reason: str = "", latency_ms: float = 0.0) -> bool:
        """Record that a backend actually placed a region in ``tier``.

        Returns False when the region cannot be resident there (larger than the
        tier's whole budget); in that case the prior placement is left intact.
        """
        region = self.regions[region_id]
        cache = self._cache_for(tier)
        if cache is None:
            self.record_evicted(region_id, reason=reason or "demoted-to-nvme")
            return True
        source = self.tiers.get(region_id, MemoryTier.NVME)
        if not cache.fits(region.size_bytes):
            self.record_event(ResidencyEvent(monotonic(), region_id, "reject", source, tier, 0, 0.0, "exceeds-tier-capacity"))
            return False
        already_there = self.states.get(region_id) == ResidencyState.RESIDENT and source == tier
        other = self._cache_for(source) if source != tier else None
        if other is not None:
            other.remove(region_id)
        evicted = cache.put(CacheEntry(region, tier, region.size_bytes))
        self.states[region_id] = ResidencyState.RESIDENT
        self.tiers[region_id] = tier
        self.record_event(ResidencyEvent(monotonic(), region_id, "resident", source, tier, 0 if already_there else region.size_bytes, latency_ms, reason))
        for victim in evicted:
            self.record_evicted(victim.region.region_id, reason="capacity")
        return True
    def record_evicted(self, region_id: str, reason: str = "") -> None:
        tier = self.tiers.get(region_id)
        was_resident = self.states.get(region_id) == ResidencyState.RESIDENT
        cache = self._cache_for(tier) if tier is not None else None
        if cache is not None: cache.remove(region_id)
        self.states[region_id] = ResidencyState.COLD
        self.tiers[region_id] = MemoryTier.NVME
        size = self.regions[region_id].size_bytes if was_resident else 0
        self.record_event(ResidencyEvent(monotonic(),region_id,"evict",tier,MemoryTier.NVME,size,0.0,reason))
    def snapshot(self) -> ResidencySnapshot:
        bytes_by_tier = {MemoryTier.VRAM:0,MemoryTier.RAM:0,MemoryTier.NVME:0}
        for rid,state in self.states.items():
            if state == ResidencyState.RESIDENT: bytes_by_tier[self.tiers[rid]] += self.regions[rid].size_bytes
            else: bytes_by_tier[MemoryTier.NVME] += self.regions[rid].size_bytes
        return ResidencySnapshot(states=dict(self.states),tiers=dict(self.tiers),scores=dict(self.scores),bytes_by_tier=bytes_by_tier)
