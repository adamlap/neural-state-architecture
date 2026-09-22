"""Residency policy: decide what should be physically resident."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from nsa.residency.types import MemoryTier, NeuralRegion

@dataclass(frozen=True)
class ResidencyDecision:
    region_id: str
    score: float
    desired_tier: MemoryTier
    retain: bool
    prefetch: bool
    reason: str

@dataclass(frozen=True)
class ResidencyPolicy:
    vram_budget_bytes: int
    ram_budget_bytes: int
    min_relevance: float = 0.05
    prefetch_threshold: float = 0.55
    retain_threshold: float = 0.35
    def decide(self, region: NeuralRegion, relevance: Optional[float], future_probability: float, dependency_probability: float = 0.0, load_latency_ms: float = 0.0) -> ResidencyDecision:
        # relevance=None means "no semantic evidence" (no state tags / untagged region).
        # That is not the same as "irrelevant": the remaining signals are renormalised
        # so a certain transition can still reach the prefetch threshold.
        known_relevance = relevance is not None
        relevance = max(0.0, min(1.0, relevance)) if known_relevance else 0.0
        future_probability = max(0.0, min(1.0, future_probability))
        dependency_probability = max(0.0, min(1.0, dependency_probability))
        latency_penalty = min(0.25, load_latency_ms / 1000.0)
        size_penalty = min(0.25, region.size_bytes / max(self.vram_budget_bytes, 1))
        if known_relevance:
            evidence = 0.30 * relevance + 0.60 * future_probability + 0.10 * dependency_probability
        else:
            evidence = (0.60 * future_probability + 0.10 * dependency_probability) / 0.70
        score = evidence - latency_penalty - size_penalty
        if score >= self.prefetch_threshold:
            tier = MemoryTier.VRAM
        elif score >= self.retain_threshold:
            tier = MemoryTier.RAM
        else:
            tier = MemoryTier.NVME
        return ResidencyDecision(region.region_id, score, tier, score >= self.retain_threshold, score >= self.prefetch_threshold, f"relevance={relevance:.3f}{'' if known_relevance else '(unknown)'}, future={future_probability:.3f}, dependency={dependency_probability:.3f}")
