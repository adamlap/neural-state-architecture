"""Durable trajectory journaling, integrity verification and replay."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from nsa.cce.events import CognitiveEvent
from nsa.cce.trajectory import CognitiveTrajectory, TrajectoryRecord
from nsa.core.state import CanonicalState
from nsa.core.state_codec import decode_state, encode_state
from nsa.core.transition import TransitionReceipt, state_digest


def _jsonable(value):
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try: return value.value
        except Exception: pass
    if isinstance(value, (tuple, list)): return [_jsonable(v) for v in value]
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"): return _jsonable(value.__dict__)
    return value


def record_to_dict(record: TrajectoryRecord, *, state: CanonicalState | None = None) -> dict:
    payload = {
        "journal_schema": "nsa.trajectory.v2",
        "step": record.step,
        "state_digest": record.state_digest,
        "receipt": _jsonable(asdict(record.receipt)) if record.receipt else None,
        "events": [
            {"kind": event.kind.value, "step": event.step, "event_id": event.event_id,
             "payload": _jsonable(event.payload), "source": event.source, "timestamp": event.timestamp}
            for event in record.events
        ],
    }
    if state is not None:
        if state.step != record.step or state_digest(state) != record.state_digest:
            raise ValueError("supplied state does not match trajectory record")
        payload["canonical_state"] = encode_state(state)
    return payload


class TrajectoryJournal:
    """Append-only JSONL journal with optional full-state snapshots.

    Version 2 records include complete canonical state material, enabling
    deterministic reconstruction instead of relying on state summaries.
    """
    def __init__(self, path: str | Path, *, persist_state: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_state = persist_state

    def append(self, record: TrajectoryRecord, *, state: CanonicalState | None = None) -> None:
        payload = record_to_dict(record, state=state if self.persist_state else None)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
            handle.flush()

    def records(self) -> tuple[dict, ...]:
        if not self.path.exists(): return ()
        result = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip(): continue
                try: result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid trajectory journal line {line_number}: {exc}") from exc
        return tuple(result)

    def verify(self, state: Optional[CanonicalState] = None) -> tuple[bool, str]:
        records = self.records()
        previous_step = -1
        for index, record in enumerate(records):
            step, digest = record.get("step"), record.get("state_digest")
            if not isinstance(step, int) or step < previous_step:
                return False, f"trajectory step ordering violated at record {index}"
            if not isinstance(digest, str) or len(digest) != 64:
                return False, f"invalid state digest at record {index}"
            snapshot = record.get("canonical_state")
            if snapshot is not None:
                try:
                    restored = decode_state(snapshot)
                except Exception as exc:
                    return False, f"state snapshot invalid at record {index}: {exc}"
                if restored.step != step or state_digest(restored) != digest:
                    return False, f"state snapshot digest mismatch at record {index}"
            previous_step = step
        if state is not None and records and records[-1]["state_digest"] != state_digest(state):
            return False, "latest journal digest does not match supplied canonical state"
        return True, "trajectory journal verified"

    def replay_states(self) -> tuple[CanonicalState, ...]:
        states = []
        for index, record in enumerate(self.records()):
            snapshot = record.get("canonical_state")
            if snapshot is None:
                raise ValueError(f"record {index} has no canonical state snapshot")
            states.append(decode_state(snapshot))
        return tuple(states)

    def append_trajectory(self, trajectory: CognitiveTrajectory) -> None:
        for record in trajectory.records:
            self.append(record)


__all__ = ["TrajectoryJournal", "record_to_dict"]
