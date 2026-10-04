from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from nsa.core.state import CanonicalState

from .config import StrategicConfig
from .engine import StrategicScenarioEngine
from .feedback import StrategicFeedback, StrategicFeedbackUpdater
from .model import StrategyCandidate, StrategyDistribution, StrategicField
from .modulation import StrategicAttentionModulator


@dataclass(frozen=True)
class StrategicEvaluation:
    distribution: StrategyDistribution
    field: StrategicField


class StrategicController:
    """Coordinate SNM while preserving an explicit enable/disable boundary."""

    def __init__(self, engine=None, modulator=None, config: StrategicConfig | None = None):
        self.config = config or StrategicConfig()
        self.engine = engine or StrategicScenarioEngine()
        self.modulator = modulator or StrategicAttentionModulator(self.config.max_bias)

    def evaluate(
        self,
        state: CanonicalState,
        candidates: Sequence[StrategyCandidate],
        *,
        urgency=None,
        risk_sensitivity=.5,
        horizon=.5,
    ) -> StrategicEvaluation:
        if not self.config.enabled:
            distribution = StrategyDistribution({}, "", 0.0, 0.0)
            field = StrategicField({}, None, 0.0, 0.0, risk_sensitivity, horizon, state.step)
            return StrategicEvaluation(distribution, field)

        d = self.engine.evaluate(candidates)
        u = max(state.soft.risk, state.soft.uncertainty) if urgency is None else urgency
        if not 0 <= u <= 1:
            raise ValueError("urgency must be in [0, 1]")
        f = StrategicField(
            d.probabilities, d.selected, d.confidence, u, risk_sensitivity, horizon, state.step
        )
        return StrategicEvaluation(d, f)

    def update_candidates(
        self,
        candidates: Sequence[StrategyCandidate],
        feedback: StrategicFeedback,
        updater: StrategicFeedbackUpdater | None = None,
    ) -> tuple[StrategyCandidate, ...]:
        if not self.config.enabled or not self.config.feedback_enabled:
            return tuple(candidates)
        return (updater or StrategicFeedbackUpdater()).update(candidates, feedback)

    def attention_bias(
        self,
        evaluation: StrategicEvaluation,
        affinity: Sequence[Mapping[str, float]],
    ) -> tuple[float, ...]:
        if not self.config.enabled or not self.config.attention_enabled:
            return tuple(0.0 for _ in affinity)
        return self.modulator.token_bias(evaluation.field, affinity)
