"""Typed events emitted by the continuous cognitive engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Mapping, Optional


class EventKind(str, Enum):
    OBSERVATION = "observation"
    BELIEF = "belief"
    PREDICTION = "prediction"
    PREDICTION_ERROR = "prediction_error"
    COGNITION = "cognition"
    GOAL = "goal"
    ACTION_PROPOSAL = "action_proposal"
    POLICY = "policy"
    CAPABILITY = "capability"
    EXECUTION = "execution"
    STATE_COMMIT = "state_commit"
    ROLLBACK = "rollback"
    ERROR = "error"


@dataclass(frozen=True)
class CognitiveEvent:
    kind: EventKind
    step: int
    event_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not self.event_id:
            raise ValueError("event_id must be non-empty")


__all__ = ["CognitiveEvent", "EventKind"]
