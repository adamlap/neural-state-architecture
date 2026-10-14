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
    # memory_id set, maintained incrementally so write() does not rescan every item
    # in Python. Derived from ``items``; excluded from equality, repr and init.
    _ids: frozenset = field(default=None, init=False, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._ids is None:
            object.__setattr__(self, "_ids", frozenset(item.memory_id for item in self.items))

    def write(self, item: MemoryItem) -> "MemoryStore":
        if item.memory_id in self._ids:
            raise ValueError(f"duplicate memory_id: {item.memory_id}")
        # Build the successor directly so the id set is extended (a C-level copy)
        # rather than recomputed by a Python-level pass over every item.
        store = object.__new__(MemoryStore)
        object.__setattr__(store, "items", self.items + (item,))
        object.__setattr__(store, "_ids", self._ids | {item.memory_id})
        return store

    def active(self, now: datetime | None = None) -> Tuple[MemoryItem, ...]:
        return tuple(item for item in self.items if item.available(now))

    def get(self, memory_id: str) -> MemoryItem:
        for item in self.items:
            if item.memory_id == memory_id:
                return item
        raise KeyError(memory_id)


__all__ = ["MemoryItem", "MemoryStore"]
