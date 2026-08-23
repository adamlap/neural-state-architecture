"""Adaptive event/salience gating for continuous CCE activity.

The gate is deliberately model-agnostic. It converts observable changes in the
continuous substrate into a bounded salience score and an invocation decision.
It does not mutate cognitive state, call an LLM, or make authority decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SalienceObservation:
    """Observable signals used to decide whether cognition should be invoked."""

    prediction_error: float = 0.0
    state_delta: float = 0.0
    input_delta: float = 0.0
    uncertainty: float = 0.0


@dataclass(frozen=True)
class SalienceDecision:
    """Immutable salience result for one observation."""

    score: float
    threshold: float
    triggered: bool
    baseline: float


class AdaptiveSalienceGate:
    """State-free-to-call, adaptive gate for event-driven cognitive work.

    The gate maintains only a scalar activity baseline. The baseline follows
    recent salience using exponential smoothing, making the invocation threshold
    adaptive rather than a fixed event schedule. All inputs are validated and
    non-finite values fail closed.
    """

    def __init__(
        self,
        *,
        prediction_weight: float = 0.40,
        state_weight: float = 0.25,
        input_weight: float = 0.20,
        uncertainty_weight: float = 0.15,
        threshold_scale: float = 1.5,
        baseline_decay: float = 0.90,
    ) -> None:
        weights = (
            prediction_weight,
            state_weight,
            input_weight,
            uncertainty_weight,
        )
        if any(w < 0 or not isfinite(w) for w in weights) or sum(weights) <= 0:
            raise ValueError("salience weights must be finite, non-negative and non-zero")
        if threshold_scale <= 0 or not isfinite(threshold_scale):
            raise ValueError("threshold_scale must be positive and finite")
        if not 0 <= baseline_decay < 1 or not isfinite(baseline_decay):
            raise ValueError("baseline_decay must be in [0, 1)")

        self._weights = weights
        self._threshold_scale = threshold_scale
        self._baseline_decay = baseline_decay
        self._baseline = 0.0

    @property
    def baseline(self) -> float:
        return self._baseline

    @staticmethod
    def _validate(observation: SalienceObservation) -> None:
        values = (
            observation.prediction_error,
            observation.state_delta,
            observation.input_delta,
            observation.uncertainty,
        )
        if any(value < 0 or not isfinite(value) for value in values):
            raise ValueError("salience observations must be finite and non-negative")

    def observe(self, observation: SalienceObservation) -> SalienceDecision:
        """Score one observation and update the adaptive activity baseline."""
        self._validate(observation)
        weighted = sum(
            value * weight
            for value, weight in zip(
                (
                    observation.prediction_error,
                    observation.state_delta,
                    observation.input_delta,
                    observation.uncertainty,
                ),
                self._weights,
            )
        ) / sum(self._weights)

        threshold = max(self._baseline * self._threshold_scale, 1e-12)
        triggered = weighted > threshold
        self._baseline = (
            self._baseline_decay * self._baseline
            + (1.0 - self._baseline_decay) * weighted
        )
        return SalienceDecision(weighted, threshold, triggered, self._baseline)


__all__ = ["AdaptiveSalienceGate", "SalienceDecision", "SalienceObservation"]
