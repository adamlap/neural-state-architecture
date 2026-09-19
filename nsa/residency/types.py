"""Typed primitives for neural virtual memory."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Dict, Optional

class MemoryTier(str, Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"

class ResidencyState(str, Enum):
    COLD = "cold"
    LOADING = "loading"
    RESIDENT = "resident"
    EVICTING = "evicting"

@dataclass(frozen=True)
class NeuralRegion:
    region_id: str
    parameter_prefixes: tuple[str, ...] = ()
    size_bytes: int = 0
    layer_index: Optional[int] = None
    semantic_tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

@dataclass(frozen=True)
class ResidencyEvent:
    timestamp: float
    region_id: str
    action: str
    source: Optional[MemoryTier]
    destination: Optional[MemoryTier]
    bytes_moved: int = 0
    latency_ms: float = 0.0
    reason: str = ""

@dataclass
class ResidencySnapshot:
    timestamp: float = field(default_factory=monotonic)
    states: Dict[str, ResidencyState] = field(default_factory=dict)
    tiers: Dict[str, MemoryTier] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    bytes_by_tier: Dict[MemoryTier, int] = field(default_factory=dict)
    def resident_regions(self, tier: Optional[MemoryTier] = None) -> list[str]:
        return [rid for rid,state in self.states.items() if state == ResidencyState.RESIDENT and (tier is None or self.tiers.get(rid) == tier)]
