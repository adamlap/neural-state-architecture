"""Bounded telemetry for neural residency decisions and transfers.

The trace is deliberately observational: it records runtime facts and never
changes NSA policy, authority, or cognitive state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

from nsa.residency.types import MemoryTier, ResidencyEvent


class ResidencyTrace:
    """Thread-safe bounded event recorder with optional JSONL persistence."""

    def __init__(self, max_events: int = 10_000, jsonl_path: Optional[str] = None) -> None:
        self.max_events = max(1, max_events)
        self._events: deque[ResidencyEvent] = deque(maxlen=self.max_events)
        self._lock = Lock()
        self.jsonl_path = Path(jsonl_path).expanduser() if jsonl_path else None

    def record(self, event: ResidencyEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(event), default=self._json_default) + "\n")

    def extend(self, events: Iterable[ResidencyEvent]) -> None:
        for event in events:
            self.record(event)

    def events(self) -> tuple[ResidencyEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def metrics(self) -> dict[str, float | int]:
        events = self.events()
        prefetch = [e for e in events if e.action == "prefetch"]
        execute = [e for e in events if e.action == "execute"]
        transfers = [e for e in events if e.action in {"resident", "evict", "prefetch", "prefetch-complete"}]
        completed = {e.region_id: e.timestamp for e in events if e.action == "prefetch-complete"}
        hits = sum(1 for e in execute if completed.get(e.region_id, -1) <= e.timestamp)
        bytes_moved = sum(e.bytes_moved for e in transfers)
        return {
            "events": len(events),
            "prefetches": len(prefetch),
            "prefetch_hits": hits,
            "prefetch_hit_rate": hits / len(prefetch) if prefetch else 0.0,
            "executions": len(execute),
            "transfer_events": len(transfers),
            "bytes_moved": bytes_moved,
            "latency_ms_total": sum(e.latency_ms for e in events),
            "latency_ms_avg": sum(e.latency_ms for e in events) / len(events) if events else 0.0,
        }

    def by_tier(self, tier: MemoryTier) -> tuple[ResidencyEvent, ...]:
        return tuple(
            event for event in self.events()
            if event.source == tier or event.destination == tier
        )

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, (MemoryTier,)):
            return value.value
        raise TypeError(f"Unsupported trace value: {type(value).__name__}")
