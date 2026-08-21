"""Persistent CCE state independent of NSA implementation internals."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import time

from ..dynamics import DynamicalState


@dataclass
class CCEState:
    tick_count: int = 0
    last_input: str | None = None
    last_reasoning: str = ""
    memories: list[str] = field(default_factory=list)
    dynamic: DynamicalState = field(default_factory=DynamicalState)
    self_confidence: float = 0.5
    self_uncertainty: float = 0.5
    started_at: float = field(default_factory=time)

    def tick(self) -> None:
        self.tick_count += 1
        self.dynamic.tick(drive=0.0)

    def observe(self, event: str) -> None:
        self.last_input = event
        self.memories.append(event)
        if len(self.memories) > 256:
            del self.memories[:-256]
