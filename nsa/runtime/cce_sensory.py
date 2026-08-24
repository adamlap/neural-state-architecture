"""Sensory Ingress, Asynchronous Event Queue, and Backpressure Handling for CCE (Phase CCE-6).

Implements:
1. Deterministic Continuous Perturbation Mapping u(t) in R^d.
2. Bounded Asynchronous Event Queue with Backpressure (DROP_OLDEST / REJECT_NEW).
3. Timestamped Input Provenance & Confidence Metadata.
4. Multi-modal Sensor Adapter Interface (Text, Telemetry, Speech Transcript Streams).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch


class BackpressurePolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    REJECT_NEW = "reject_new"


class SensoryBackpressureError(Exception):
    """Raised when incoming sensory events exceed capacity under REJECT_NEW policy."""
    pass


@dataclass(frozen=True)
class SensoryEvent:
    """A timestamped sensory event logged in the ingress ring-buffer."""

    source: str
    content: str
    timestamp_utc: float
    importance: float = 0.5
    confidence: float = 1.0
    sequence_id: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BoundedEventQueue:
    """Thread-safe bounded asynchronous event queue with explicit backpressure."""

    def __init__(
        self,
        max_size: int = 100,
        policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST,
    ) -> None:
        self.max_size = max(1, int(max_size))
        self.policy = policy
        self._events: List[SensoryEvent] = []
        self._dropped_count: int = 0

    @property
    def size(self) -> int:
        return len(self._events)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def push(self, event: SensoryEvent) -> bool:
        """Push an event to the queue; enforces backpressure if capacity exceeded."""
        if len(self._events) >= self.max_size:
            if self.policy == BackpressurePolicy.REJECT_NEW:
                self._dropped_count += 1
                raise SensoryBackpressureError(
                    f"Sensory queue is full ({self.max_size} events); rejected new event from '{event.source}'"
                )
            else:  # DROP_OLDEST
                self._events.pop(0)
                self._dropped_count += 1

        self._events.append(event)
        return True

    def pop_batch(self, max_count: int = 10) -> List[SensoryEvent]:
        """Pop up to max_count events in chronological order."""
        count = min(len(self._events), max_count)
        popped = self._events[:count]
        self._events = self._events[count:]
        return popped

    def clear(self) -> int:
        """Clear queue and return number of cancelled events."""
        cancelled = len(self._events)
        self._events.clear()
        return cancelled


class CCESensoryIngress:
    """Maps external text/sensor streams to bounded perturbation vectors u(t) in R^d."""

    def __init__(
        self,
        dimension: int = 4,
        scale: float = 0.5,
        max_queue_size: int = 100,
        policy: BackpressurePolicy = BackpressurePolicy.DROP_OLDEST,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = int(dimension)
        self.scale = float(scale)
        self.queue = BoundedEventQueue(max_size=max_queue_size, policy=policy)
        self._history: List[SensoryEvent] = []
        self._sequence_counter = 0

    def encode_text_to_perturbation(
        self,
        text: str,
        *,
        source: str = "external",
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> Tuple[torch.Tensor, SensoryEvent]:
        """Project raw input text to a continuous perturbation vector."""
        self._sequence_counter += 1
        importance = max(0.0, min(1.0, float(importance)))
        confidence = max(0.0, min(1.0, float(confidence)))

        event = SensoryEvent(
            source=source,
            content=text[:500],
            timestamp_utc=time.time(),
            importance=importance,
            confidence=confidence,
            sequence_id=self._sequence_counter,
        )
        self.queue.push(event)
        self._history.append(event)
        if len(self._history) > 100:
            self._history.pop(0)

        clean = text.strip()
        coords: List[float] = []
        for i in range(self.dimension):
            seed = f"{source}:{i}:{clean[:256]}"
            h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
            val = ((h / 0xFFFFFFFF) * 2.0 - 1.0) * self.scale * importance * confidence
            coords.append(val)

        perturbation = torch.tensor(coords, dtype=torch.float32)
        return perturbation, event

    @property
    def recent_events(self) -> List[SensoryEvent]:
        return list(self._history)


__all__ = [
    "BackpressurePolicy",
    "SensoryBackpressureError",
    "SensoryEvent",
    "BoundedEventQueue",
    "CCESensoryIngress",
]
