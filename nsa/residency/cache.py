"""Bounded residency cache with explicit byte accounting."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from nsa.residency.types import MemoryTier, NeuralRegion

@dataclass
class CacheEntry:
    region: NeuralRegion
    tier: MemoryTier
    bytes_resident: int

class ResidencyCache:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = max(0, capacity_bytes)
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._bytes = 0
    @property
    def bytes_used(self) -> int: return self._bytes
    def contains(self, region_id: str) -> bool: return region_id in self._entries
    def touch(self, region_id: str) -> None:
        if region_id in self._entries: self._entries.move_to_end(region_id)
    def put(self, entry: CacheEntry) -> list[CacheEntry]:
        evicted = []
        old = self._entries.pop(entry.region.region_id, None)
        if old: self._bytes -= old.bytes_resident
        self._entries[entry.region.region_id] = entry
        self._bytes += entry.bytes_resident
        while self._bytes > self.capacity_bytes and self._entries:
            rid,victim = self._entries.popitem(last=False)
            self._bytes -= victim.bytes_resident
            if rid != entry.region.region_id: evicted.append(victim)
            else: break
        return evicted
    def remove(self, region_id: str) -> CacheEntry | None:
        entry = self._entries.pop(region_id, None)
        if entry: self._bytes -= entry.bytes_resident
        return entry
    def entries(self) -> tuple[CacheEntry, ...]: return tuple(self._entries.values())
