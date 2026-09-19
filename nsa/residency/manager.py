"""Neural virtual-memory manager."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping, Sequence
from nsa.residency.cache import CacheEntry, ResidencyCache
from nsa.residency.policy import ResidencyDecision, ResidencyPolicy
from nsa.residency.predictor import HeuristicResidencyPredictor, ResidencyPredictor
from nsa.residency.types import MemoryTier, NeuralRegion, ResidencyEvent, ResidencySnapshot, ResidencyState

@dataclass
class NeuralResidencyManager:
    policy: ResidencyPolicy
    predictor: ResidencyPredictor = field(default_factory=HeuristicResidencyPredictor)
    trace: Any = field(default=None, repr=False)
    def __post_init__(self) -> None:
        self.regions: dict[str, NeuralRegion] = {}
        self.states: dict[str, ResidencyState] = {}
        self.tiers: dict[str, MemoryTier] = {}
        self.scores: dict[str, float] = {}
        self.events: list[ResidencyEvent] = []
        self.vram_cache = ResidencyCache(self.policy.vram_budget_bytes)
        self.ram_cache = ResidencyCache(self.policy.ram_budget_bytes)
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
    def _relevance(self, region: NeuralRegion, state: Mapping[str, object]) -> float:
        tags = state.get("tags", ())
        if not tags: return 0.0
        tag_set = {str(t) for t in tags}
        return min(1.0, sum(t in tag_set for t in region.semantic_tags) / max(len(region.semantic_tags),1))
    def _dependency_probability(self, region: NeuralRegion, probabilities: Mapping[str,float]) -> float:
        return max((probabilities.get(dep,0.0) for dep in region.dependencies), default=0.0)
    def record_resident(self, region_id: str, tier: MemoryTier, reason: str = "") -> None:
        region = self.regions[region_id]
        source = self.tiers.get(region_id, MemoryTier.NVME)
        started = monotonic()
        cache = self.vram_cache if tier == MemoryTier.VRAM else self.ram_cache
        evicted = cache.put(CacheEntry(region,tier,region.size_bytes))
        self.states[region_id] = ResidencyState.RESIDENT
        self.tiers[region_id] = tier
        self.current_region = region_id
        self.record_event(ResidencyEvent(monotonic(),region_id,"resident",source,tier,region.size_bytes,(monotonic()-started)*1000,reason))
        for victim in evicted: self.record_evicted(victim.region.region_id, reason="capacity")
    def record_evicted(self, region_id: str, reason: str = "") -> None:
        tier = self.tiers.get(region_id)
        if tier == MemoryTier.VRAM: self.vram_cache.remove(region_id)
        elif tier == MemoryTier.RAM: self.ram_cache.remove(region_id)
        self.states[region_id] = ResidencyState.COLD
        self.tiers[region_id] = MemoryTier.NVME
        self.record_event(ResidencyEvent(monotonic(),region_id,"evict",tier,MemoryTier.NVME,self.regions[region_id].size_bytes,0.0,reason))
    def snapshot(self) -> ResidencySnapshot:
        bytes_by_tier = {MemoryTier.VRAM:0,MemoryTier.RAM:0,MemoryTier.NVME:0}
        for rid,state in self.states.items():
            if state == ResidencyState.RESIDENT: bytes_by_tier[self.tiers[rid]] += self.regions[rid].size_bytes
            else: bytes_by_tier[MemoryTier.NVME] += self.regions[rid].size_bytes
        return ResidencySnapshot(states=dict(self.states),tiers=dict(self.tiers),scores=dict(self.scores),bytes_by_tier=bytes_by_tier)
