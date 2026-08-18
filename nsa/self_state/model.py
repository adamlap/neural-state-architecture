"""Explicit self-state primitives for metacognition experiments.

This module intentionally models *computational self-awareness* rather than
claiming consciousness. The values are observations about the system's own
state that can be consumed by a model or runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class SelfState:
    """Compact representation of the agent's current computational condition.

    All scalar fields are normalized to [0, 1]. They are descriptive signals,
    not proofs of the corresponding property.
    """

    confidence: float = 1.0
    uncertainty: float = 0.0
    perceived_risk: float = 0.0
    capability_awareness: float = 0.0
    resource_pressure: float = 0.0
    goal_progress: float = 0.0
    state_prediction_error: float = 0.0
    step: int = 0

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        for name in (
            "confidence",
            "uncertainty",
            "perceived_risk",
            "capability_awareness",
            "resource_pressure",
            "goal_progress",
            "state_prediction_error",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

    def observe(self, **updates: float) -> "SelfState":
        """Create a new self-state from runtime/model observations."""
        allowed = {
            "confidence",
            "uncertainty",
            "perceived_risk",
            "capability_awareness",
            "resource_pressure",
            "goal_progress",
            "state_prediction_error",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"unknown self-state fields: {sorted(unknown)}")
        return replace(self, **updates, step=self.step + 1)

    def metacognitive_pressure(self) -> float:
        """Return a bounded signal for how strongly self-monitoring is needed."""
        return min(
            1.0,
            max(
                self.uncertainty,
                self.perceived_risk,
                self.resource_pressure,
                self.state_prediction_error,
            ),
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "step": self.step,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "perceived_risk": self.perceived_risk,
            "capability_awareness": self.capability_awareness,
            "resource_pressure": self.resource_pressure,
            "goal_progress": self.goal_progress,
            "state_prediction_error": self.state_prediction_error,
            "metacognitive_pressure": self.metacognitive_pressure(),
        }


@dataclass(frozen=True)
class SelfStateObservation:
    """An observation that can update self-state without granting authority."""

    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    perceived_risk: Optional[float] = None
    capability_awareness: Optional[float] = None
    resource_pressure: Optional[float] = None
    goal_progress: Optional[float] = None
    state_prediction_error: Optional[float] = None

    def apply(self, state: SelfState) -> SelfState:
        updates = {
            name: value
            for name, value in {
                "confidence": self.confidence,
                "uncertainty": self.uncertainty,
                "perceived_risk": self.perceived_risk,
                "capability_awareness": self.capability_awareness,
                "resource_pressure": self.resource_pressure,
                "goal_progress": self.goal_progress,
                "state_prediction_error": self.state_prediction_error,
            }.items()
            if value is not None
        }
        return state.observe(**updates)


__all__ = ["SelfState", "SelfStateObservation"]
