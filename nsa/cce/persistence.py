"""Durable trajectory journaling and integrity verification."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.trajectory import CognitiveTrajectory, TrajectoryRecord
from nsa.core.state import CanonicalState
from nsa.core.transition import TransitionReceipt, state_digest


def _jsonable(value):
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return value.value
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return value


def record_to_dict(record: TrajectoryRecord) -> dict:
    return {
        "step": record.step,
        "state_digest": record.state_digest,
        "receipt": _jsonable(asdict(record.receipt)) if record.receipt else None,
        "events": [
            {"kind": event.kind.value, "step": event.step, "event_id": event.event_id,
             "payload": _jsonable(event.payload), "source": event.source, "timestamp": event.timestamp}
            for event in record.events
        ],
    }


class TrajectoryJournal:
    """Append-only JSONL journal for cognitive trajectory records.

    The journal never mutates prior records. ``verify`` checks ordering and,
    when a canonical state is supplied, verifies its digest against the latest
    record. Storage is deliberately dumb so durable backends can replace it.
    """
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: TrajectoryRecord) -> None:
        payload = record_to_dict(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()

    def records(self) -> tuple[dict, ...]:
        if not self.path.exists():
            return ()
        result = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid trajectory journal line {line_number}: {exc}") from exc
        return tuple(result)

    def verify(self, state: Optional[CanonicalState] = None) -> tuple[bool, str]:
        records = self.records()
        previous_step = -1
        for index, record in enumerate(records):
            step = record.get("step")
            digest = record.get("state_digest")
            if not isinstance(step, int) or step < previous_step:
                return False, f"trajectory step ordering violated at record {index}"
            if not isinstance(digest, str) or len(digest) != 64:
                return False, f"invalid state digest at record {index}"
            previous_step = step
        if state is not None and records and records[-1]["state_digest"] != state_digest(state):
            return False, "latest journal digest does not match supplied canonical state"
        return True, "trajectory journal verified"

    def append_trajectory(self, trajectory: CognitiveTrajectory) -> None:
        for record in trajectory.records:
            self.append(record)


__all__ = ["TrajectoryJournal", "record_to_dict"]
