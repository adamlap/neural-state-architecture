"""Durable lifecycle primitives for the CCE runtime.

The lifecycle layer is intentionally independent of the LLM backend. It owns
checkpoint metadata, explicit reset semantics, and asynchronous input events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CognitiveInputEvent:
    payload: Any
    source: str = "text"
    confidence: float = 1.0
    timestamp: str = ""
    provenance: str = "local"

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(self, "timestamp", _now())
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class CheckpointEnvelope:
    schema_version: int
    created_at: str
    state: dict[str, Any]
    state_hash: str

    @staticmethod
    def hash_state(state: dict[str, Any]) -> str:
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def create(cls, state: dict[str, Any], schema_version: int = 1) -> "CheckpointEnvelope":
        return cls(schema_version, _now(), state, cls.hash_state(state))


class StateCheckpointStore:
    """Atomic JSON checkpoint store with integrity verification."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, state: dict[str, Any]) -> CheckpointEnvelope:
        envelope = CheckpointEnvelope.create(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # unique temp file: concurrent saves must not share (and corrupt) one ".tmp" name
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=str(self.path.parent), suffix=".tmp", delete=False, encoding="utf-8") as handle:
                temporary = handle.name
                handle.write(json.dumps(asdict(envelope), indent=2, sort_keys=True))
                handle.flush()
                os.fsync(handle.fileno())  # durable before the rename publishes it
            os.replace(temporary, self.path)
        except BaseException:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
            raise
        return envelope

    def load(self) -> CheckpointEnvelope:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        envelope = CheckpointEnvelope(**raw)
        expected = CheckpointEnvelope.hash_state(envelope.state)
        if expected != envelope.state_hash:
            raise ValueError("checkpoint integrity hash mismatch")
        return envelope

    def exists(self) -> bool:
        return self.path.exists()


class CognitiveInputQueue:
    """Small FIFO event boundary for asynchronous sensory/text adapters."""

    def __init__(self) -> None:
        self._events: list[CognitiveInputEvent] = []

    def push(self, event: CognitiveInputEvent) -> None:
        self._events.append(event)

    def drain(self) -> list[CognitiveInputEvent]:
        events, self._events = self._events, []
        return events

    def __len__(self) -> int:
        return len(self._events)
