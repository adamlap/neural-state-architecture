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
    def fits(self, size_bytes: int) -> bool:
        """Whether an entry of this size could ever be resident in this cache."""
        return size_bytes <= self.capacity_bytes
    def touch(self, region_id: str) -> None:
        if region_id in self._entries: self._entries.move_to_end(region_id)
    def put(self, entry: CacheEntry) -> list[CacheEntry]:
        """Insert an entry, evicting least-recently-used entries to make room.

        An entry larger than the whole cache is rejected: it is returned as the
        sole "evicted" item and the existing contents are left untouched, rather
        than flushing the cache for something that can never be resident.
        """
        region_id = entry.region.region_id
        old = self._entries.pop(region_id, None)
        if old:
            self._bytes -= old.bytes_resident
        if not self.fits(entry.bytes_resident):
            return [entry]
        evicted: list[CacheEntry] = []
        while self._bytes + entry.bytes_resident > self.capacity_bytes and self._entries:
            _, victim = self._entries.popitem(last=False)
            self._bytes -= victim.bytes_resident
            evicted.append(victim)
        self._entries[region_id] = entry
        self._bytes += entry.bytes_resident
        return evicted
    def remove(self, region_id: str) -> CacheEntry | None:
        entry = self._entries.pop(region_id, None)
        if entry: self._bytes -= entry.bytes_resident
        return entry
    def entries(self) -> tuple[CacheEntry, ...]: return tuple(self._entries.values())
