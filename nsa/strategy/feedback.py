from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nsa.cognition.interfaces import PredictionError
from .model import StrategyCandidate


@dataclass(frozen=True)
class StrategicFeedback:
    """Observed evidence used to update soft strategic priors."""
    strategy_id: str
    prediction_error: PredictionError
    outcome_id: str | None = None
    observed_goal_progress: float | None = None
    observed_risk: float | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        for name, value in (
            ("observed_goal_progress", self.observed_goal_progress),
            ("observed_risk", self.observed_risk),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class StrategicFeedbackUpdater:
    """Apply bounded prediction-error feedback to strategic priors.

    Feedback only changes the next strategic distribution's soft prior. It
    cannot change capabilities, canonical state, governance, or the action set.
    """
    def __init__(
        self,
        *,
        learning_rate: float = 0.25,
        max_delta: float = 0.20,
        min_prior: float = 0.05,
        max_prior: float = 1.0,
    ) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 < max_delta <= 1.0:
            raise ValueError("max_delta must be in (0, 1]")
        if not 0.0 <= min_prior <= max_prior <= 1.0:
            raise ValueError("prior bounds must satisfy 0 <= min <= max <= 1")
        self.learning_rate = learning_rate
        self.max_delta = max_delta
        self.min_prior = min_prior
        self.max_prior = max_prior

    def update(
        self,
        candidates: Sequence[StrategyCandidate],
        feedback: StrategicFeedback,
    ) -> tuple[StrategyCandidate, ...]:
        error = min(1.0, feedback.prediction_error.magnitude)
        if error == 0.0:
            return tuple(candidates)
        if feedback.observed_goal_progress is None and feedback.observed_risk is None:
            return tuple(candidates)

        progress = feedback.observed_goal_progress or 0.0
        risk = feedback.observed_risk or 0.0
        quality = progress - risk
        delta = min(self.max_delta, self.learning_rate * error)
        signed_delta = delta if quality >= 0.0 else -delta

        updated: list[StrategyCandidate] = []
        for candidate in candidates:
            if candidate.strategy_id != feedback.strategy_id:
                updated.append(candidate)
                continue
            prior = min(self.max_prior, max(self.min_prior, candidate.prior + signed_delta))
            updated.append(
                StrategyCandidate(
                    strategy_id=candidate.strategy_id,
                    outcomes=candidate.outcomes,
                    prior=prior,
                    information_gain=candidate.information_gain,
                    reversibility=candidate.reversibility,
                )
            )
        return tuple(updated)
