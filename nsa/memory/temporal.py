"""Temporal canonical memory with explicit versioning and supersession.

Episodic memory remains append-only, while canonical retrieval exposes only the
latest valid version for a logical memory key unless historical retrieval is
explicitly requested.

"Latest" means latest *written*. That is last-write-wins and it is deliberate, but
it has a sharp edge: a later low-authority assertion of an old value silently
becomes CURRENT. Each version therefore records, in its content, whether it
re-asserts a previously superseded value (``reasserts_version``) and whether it
overrides an explicit update with a plain fact (``overrides_update``);
``conflicts()`` returns the keys where that happened so callers can decide what
to do instead of discovering it from a wrong answer.

Integrity gate: every write carries an ``IntegrityLabel`` (``TRUSTED`` by default). An
``UNTRUSTED`` write may *not* supersede a ``TRUSTED`` current value. It is appended to the
episodic store for audit with ``quarantined=True`` but never enters the canonical version
chain, so no retrieval path (current, history, render) can return it. A ``TRUSTED`` write
always supersedes. The label is supplied by the caller out of band (the ingress path that
knows where the bytes came from); text inside the content cannot raise it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from nsa.algebra import IntegrityLabel

from .model import MemoryItem, MemoryStore


def _norm(key: Any) -> str:
    return str(key).strip().lower()


@dataclass(frozen=True)
class TemporalMemoryStore:
    """Immutable append-only memory with canonical latest-value retrieval."""

    store: MemoryStore = MemoryStore()
    # normalised key -> that key's versions in write order. Derived from ``store``
    # and maintained incrementally so lookups do not rescan the whole store.
    _index: Mapping[str, tuple] = field(default=None, init=False, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._index is None:
            index: dict[str, list[MemoryItem]] = {}
            for item in self.store.items:
                content = item.content
                if isinstance(content, Mapping) and "memory_key" in content and not content.get("quarantined"):
                    index.setdefault(_norm(content["memory_key"]), []).append(item)
            object.__setattr__(self, "_index", {k: tuple(v) for k, v in index.items()})

    def remember(
        self,
        key: str,
        content: Any,
        *,
        kind: str = "fact",
        provenance_ids: Sequence[str] = (),
        sensitivity: str = "normal",
        created_at: datetime | None = None,
        integrity: IntegrityLabel = IntegrityLabel.TRUSTED,
    ) -> "TemporalMemoryStore":
        key = str(key).strip()
        if not key:
            raise ValueError("memory key must not be empty")
        chain = self._index.get(_norm(key), ())
        current = self._current_of(tuple(i for i in chain if i.available()))
        ordinal = len(self.store.items)
        payload = dict(content) if isinstance(content, Mapping) else {"value": content}
        version = int(current.content.get("version", 0)) + 1 if current else 1
        payload.update({"integrity": integrity.name})
        if (
            current is not None
            and integrity is IntegrityLabel.UNTRUSTED
            and current.content.get("integrity", IntegrityLabel.TRUSTED.name) == IntegrityLabel.TRUSTED.name
        ):
            # Audit-only: kept in the episodic store, absent from the canonical index.
            payload.update({"memory_key": key, "quarantined": True, "blocked_version": int(current.content.get("version", 0))})
            item = MemoryItem(
                memory_id=f"tm-{ordinal:08d}", content=payload, kind=kind,
                provenance_ids=tuple(provenance_ids), sensitivity=sensitivity,
                created_at=created_at or datetime.now(timezone.utc),
            )
            successor = object.__new__(TemporalMemoryStore)
            object.__setattr__(successor, "store", self.store.write(item))
            object.__setattr__(successor, "_index", self._index)
            return successor
        payload.update({
            "memory_key": key,
            "version": version,
            "supersedes_id": current.memory_id if current else None,
        })
        if current is not None:
            new_value = payload.get("value")
            earlier = [i for i in chain if i is not current and self._value(i) == new_value]
            if earlier and new_value != self._value(current):
                payload["reasserts_version"] = int(earlier[-1].content.get("version", 0))
            if current.kind == "update" and kind != "update" and new_value != self._value(current):
                payload["overrides_update"] = True
        item = MemoryItem(
            memory_id=f"tm-{ordinal:08d}",
            content=payload,
            kind=kind,
            provenance_ids=tuple(provenance_ids),
            sensitivity=sensitivity,
            created_at=created_at or datetime.now(timezone.utc),
        )
        successor = object.__new__(TemporalMemoryStore)
        object.__setattr__(successor, "store", self.store.write(item))
        index = dict(self._index)
        index[_norm(key)] = chain + (item,)
        object.__setattr__(successor, "_index", index)
        return successor

    def versions(self, key: str, now: datetime | None = None) -> tuple[MemoryItem, ...]:
        return tuple(i for i in self._index.get(_norm(key), ()) if i.available(now))

    @staticmethod
    def _current_of(versions: Sequence[MemoryItem]) -> MemoryItem | None:
        return max(versions, key=lambda item: int(item.content.get("version", 0)), default=None)

    def current(self, key: str) -> MemoryItem | None:
        return self._current_of(self.versions(key))

    def quarantined(self) -> tuple[MemoryItem, ...]:
        """Writes rejected by the integrity gate, in write order (for audit)."""
        return tuple(
            i for i in self.store.items
            if isinstance(i.content, Mapping) and i.content.get("quarantined")
        )

    def conflicts(self) -> tuple[MemoryItem, ...]:
        """Versions that re-assert a superseded value or override an explicit update."""
        return tuple(
            item
            for chain in self._index.values()
            for item in chain
            if "reasserts_version" in item.content or item.content.get("overrides_update")
        )

    def retrieve(self, keys: Sequence[str], *, limit: int = 3) -> tuple[MemoryItem, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        selected = [item for key in keys if (item := self.current(key)) is not None]
        selected.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(selected[:limit])

    def history(self, key: str, *, limit: int = 10) -> tuple[MemoryItem, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(
            sorted(
                self.versions(key),
                key=lambda item: int(item.content.get("version", 0)),
                reverse=True,
            )[:limit]
        )

    @staticmethod
    def _value(item: MemoryItem) -> Any:
        content = item.content
        if isinstance(content, Mapping):
            return content.get("value", content)
        return content

    def render(
        self,
        keys: Sequence[str],
        *,
        limit: int = 3,
        mode: str = "current",
        include_current: bool = False,
    ) -> str:
        """Render memory with explicit temporal semantics for an LLM.

        History rows are ``STATUS=SUPERSEDED`` only. Appending ``CURRENT_VALUE=`` to them
        (``include_current=True``) made qwen2.5:1.5b answer the current value to "what was
        the previous value" 18/18 times (0/18 correct); without it 18/18 correct. A model
        that needs both values should be given the CURRENT rows separately.
        """
        if mode not in {"current", "history"}:
            raise ValueError("mode must be 'current' or 'history'")

        if mode == "current":
            items = self.retrieve(keys, limit=limit)
            if not items:
                return "NO_RELEVANT_MEMORY"
            return "\n".join(
                f"- {item.content.get('memory_key')}: {self._value(item)} "
                f"VERSION={item.content.get('version')} STATUS=CURRENT"
                for item in items
            )

        rows: list[str] = []
        for key in keys:
            versions = self.history(key, limit=limit + 1)
            if not versions:
                continue
            current = versions[0]
            for item in versions[1:limit + 1]:
                row = (
                    f"- {item.content.get('memory_key')}: {self._value(item)} "
                    f"VERSION={item.content.get('version')} STATUS=SUPERSEDED"
                )
                if include_current:
                    row += f" CURRENT_VALUE={self._value(current)}"
                rows.append(row)
        return "\n".join(rows) if rows else "NO_RELEVANT_MEMORY_HISTORY"


__all__ = ["TemporalMemoryStore"]
