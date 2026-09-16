"""Append-only cognitive trajectory for observability and replay."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.core.state import CanonicalState
from nsa.core.transition import TransitionReceipt, state_digest


@dataclass(frozen=True)
class TrajectoryRecord:
    step: int
    state_digest: str
    receipt: Optional[TransitionReceipt] = None
    events: Tuple[CognitiveEvent, ...] = ()


class CognitiveTrajectory:
    """In-memory append-only trajectory; persistence can consume its records."""

    def __init__(self, initial_state: CanonicalState) -> None:
        self._records: list[TrajectoryRecord] = [
            TrajectoryRecord(initial_state.step, state_digest(initial_state))
        ]

    @property
    def records(self) -> Tuple[TrajectoryRecord, ...]:
        return tuple(self._records)

    @property
    def latest(self) -> TrajectoryRecord:
        return self._records[-1]

    def append(
        self,
        state: CanonicalState,
        *,
        receipt: Optional[TransitionReceipt] = None,
        events: Iterable[CognitiveEvent] = (),
    ) -> TrajectoryRecord:
        record = TrajectoryRecord(state.step, state_digest(state), receipt, tuple(events))
        if record.step < self.latest.step:
            raise ValueError("trajectory step cannot move backwards")
        self._records.append(record)
        return record

    def events(self, kind: Optional[EventKind] = None) -> Tuple[CognitiveEvent, ...]:
        result = [event for record in self._records for event in record.events]
        if kind is not None:
            result = [event for event in result if event.kind == kind]
        return tuple(result)

    def replay_digests(self) -> Tuple[str, ...]:
        return tuple(record.state_digest for record in self._records)

    def clear_after(self, step: int) -> None:
        if step < 0:
            raise ValueError("step must be non-negative")
        self._records = [record for record in self._records if record.step <= step]
        if not self._records:
            raise ValueError("cannot clear the entire trajectory")


__all__ = ["CognitiveTrajectory", "TrajectoryRecord"]
