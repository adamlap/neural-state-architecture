"""Predictive self-state primitives for NSA metacognition research."""

from __future__ import annotations

from dataclasses import dataclass

from .model import SelfState


@dataclass(frozen=True)
class SelfStatePrediction:
    predicted: SelfState
    observed: SelfState | None = None

    def compare(self, observed: SelfState) -> "SelfStatePrediction":
        return SelfStatePrediction(self.predicted, observed)

    def error(self, observed: SelfState | None = None) -> float:
        target = observed or self.observed
        if target is None:
            raise ValueError("an observed self-state is required")
        fields = (
            "confidence",
            "uncertainty",
            "perceived_risk",
            "capability_awareness",
            "resource_pressure",
            "goal_progress",
            "state_prediction_error",
        )
        return sum(abs(getattr(self.predicted, f) - getattr(target, f)) for f in fields) / len(fields)


class SelfStatePredictor:
    """Small baseline predictor; future versions can be learned models."""

    def predict(self, state: SelfState, **expected_updates: float) -> SelfStatePrediction:
        predicted = state.observe(**expected_updates) if expected_updates else state
        return SelfStatePrediction(predicted)


__all__ = ["SelfStatePrediction", "SelfStatePredictor"]
