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
        self._handle = None

    def record(self, event: ResidencyEvent) -> None:
        with self._lock:
            self._events.append(event)
            if self.jsonl_path is not None:
                if self._handle is None:
                    self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    self._handle = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
                self._handle.write(json.dumps(asdict(event), default=self._json_default) + "\n")

    def close(self) -> None:
        """Flush and release the JSONL handle (recording again reopens it)."""
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

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
        events = sorted(self.events(), key=lambda e: e.timestamp)
        prefetches = [e for e in events if e.action == "prefetch"]
        executions = [e for e in events if e.action == "execute"]
        completions = [e for e in events if e.action == "prefetch-complete"]
        skipped = [e for e in events if e.action == "prefetch-skipped"]
        errors = [e for e in events if e.action == "prefetch-error"]
        # Placement transfers only. Page-cache prefetch is reported separately
        # because it warms a cache; it does not move a region between tiers.
        transfers = [e for e in events if e.action in {"resident", "evict"}]
        hits = self._prefetch_hits(events)
        return {
            "events": len(events),
            "prefetches": len(prefetches),
            "prefetch_completed": len(completions),
            "prefetch_skipped": len(skipped),
            "prefetch_errors": len(errors),
            "prefetch_hits": hits,
            # fraction of issued prefetches later consumed by an execution
            "prefetch_hit_rate": min(1.0, hits / len(prefetches)) if prefetches else 0.0,
            # fraction of executions that had a completed prefetch waiting
            "prefetch_coverage": hits / len(executions) if executions else 0.0,
            "bytes_prefetched": sum(e.bytes_moved for e in completions),
            "executions": len(executions),
            "transfer_events": len(transfers),
            "bytes_moved": sum(e.bytes_moved for e in transfers),
            "latency_ms_total": sum(e.latency_ms for e in events),
            "latency_ms_avg": sum(e.latency_ms for e in events) / len(events) if events else 0.0,
        }

    @staticmethod
    def _prefetch_hits(events: list[ResidencyEvent]) -> int:
        """Count completed prefetches that finished before an execution began.

        ``execute`` events are stamped when the region finishes, so its start is
        ``timestamp - latency``. Each completed prefetch can serve at most one
        subsequent execution of the same region.
        """
        pending: dict[str, list[float]] = {}
        hits = 0
        ordered = sorted(
            (e for e in events if e.action in {"prefetch-complete", "execute"}),
            key=lambda e: e.timestamp - (e.latency_ms / 1000.0 if e.action == "execute" else 0.0),
        )
        for event in ordered:
            if event.action == "prefetch-complete":
                pending.setdefault(event.region_id, []).append(event.timestamp)
                continue
            started = event.timestamp - event.latency_ms / 1000.0
            queue = pending.get(event.region_id)
            if queue and queue[0] <= started:
                queue.pop(0)
                hits += 1
        return hits

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
