"""Temporal canonical memory with explicit versioning and supersession.

Episodic memory remains append-only, while canonical retrieval exposes only the
latest valid version for a logical memory key unless historical retrieval is
explicitly requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .model import MemoryItem, MemoryStore


@dataclass(frozen=True)
class TemporalMemoryStore:
    """Immutable append-only memory with canonical latest-value retrieval."""

    store: MemoryStore = MemoryStore()

    def remember(
        self,
        key: str,
        content: Any,
        *,
        kind: str = "fact",
        provenance_ids: Sequence[str] = (),
        sensitivity: str = "normal",
        created_at: datetime | None = None,
    ) -> "TemporalMemoryStore":
        key = str(key).strip()
        if not key:
            raise ValueError("memory key must not be empty")
        current = self.current(key)
        ordinal = len(self.store.items)
        item = MemoryItem(
            memory_id=f"tm-{ordinal:08d}",
            content=content,
            kind=kind,
            provenance_ids=tuple(provenance_ids),
            sensitivity=sensitivity,
            created_at=created_at or datetime.now(timezone.utc),
        )
        # Store temporal metadata separately on the item-compatible content so
        # existing MemoryItem consumers remain backwards compatible.
        payload = dict(content) if isinstance(content, Mapping) else {"value": content}
        payload.update({
            "memory_key": key,
            "version": (int(current.content.get("version", 0)) + 1) if current and isinstance(current.content, Mapping) else 1,
            "supersedes_id": current.memory_id if current else None,
        })
        item = MemoryItem(
            memory_id=item.memory_id,
            content=payload,
            kind=kind,
            provenance_ids=item.provenance_ids,
            sensitivity=item.sensitivity,
            created_at=item.created_at,
            expires_at=item.expires_at,
        )
        return TemporalMemoryStore(self.store.write(item))

    def versions(self, key: str) -> tuple[MemoryItem, ...]:
        key = str(key).strip()
        return tuple(
            item for item in self.store.active()
            if isinstance(item.content, Mapping)
            and str(item.content.get("memory_key", "")).lower() == key.lower()
        )

    def current(self, key: str) -> MemoryItem | None:
        versions = self.versions(key)
        return max(versions, key=lambda item: item.created_at, default=None)

    def retrieve(self, keys: Sequence[str], *, limit: int = 3) -> tuple[MemoryItem, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = [item for key in keys if (item := self.current(key)) is not None]
        selected.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(selected[:limit])

    def history(self, key: str, *, limit: int = 10) -> tuple[MemoryItem, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(sorted(self.versions(key), key=lambda item: item.created_at, reverse=True)[:limit])

    def render(self, keys: Sequence[str], *, limit: int = 3) -> str:
        items = self.retrieve(keys, limit=limit)
        if not items:
            return "NO_RELEVANT_MEMORY"
        return "\n".join(
            f"- {item.content.get('memory_key')}: {item.content.get('value', item.content)}"
            for item in items
        )


__all__ = ["TemporalMemoryStore"]
