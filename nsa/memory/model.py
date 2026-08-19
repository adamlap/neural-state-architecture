"""Typed memory with explicit provenance and state metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Tuple


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    content: Any
    kind: str
    provenance_ids: Tuple[str, ...] = ()
    sensitivity: str = "normal"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def available(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.expires_at is None or now < self.expires_at


@dataclass(frozen=True)
class MemoryStore:
    items: Tuple[MemoryItem, ...] = ()

    def write(self, item: MemoryItem) -> "MemoryStore":
        if any(existing.memory_id == item.memory_id for existing in self.items):
            raise ValueError(f"duplicate memory_id: {item.memory_id}")
        return MemoryStore(self.items + (item,))

    def active(self, now: datetime | None = None) -> Tuple[MemoryItem, ...]:
        return tuple(item for item in self.items if item.available(now))

    def get(self, memory_id: str) -> MemoryItem:
        for item in self.items:
            if item.memory_id == memory_id:
                return item
        raise KeyError(memory_id)


__all__ = ["MemoryItem", "MemoryStore"]
