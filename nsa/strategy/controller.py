"""High-level optional strategic control layer."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from nsa.core.state import CanonicalState
from .engine import StrategicScenarioEngine
from .model import StrategyCandidate, StrategyDistribution, StrategicField
from .modulation import StrategicAttentionModulator

@dataclass(frozen=True)
class StrategicEvaluation:
    distribution: StrategyDistribution
    field: StrategicField

class StrategicController:
    """Optional strategic layer; it does not replace NSA governance or cognition."""
    def __init__(self, *, engine=None, modulator=None):
        self.engine = engine or StrategicScenarioEngine()
        self.modulator = modulator or StrategicAttentionModulator()

    def evaluate(self, state: CanonicalState, candidates: Sequence[StrategyCandidate],
                 *, urgency: float | None = None, risk_sensitivity=0.5,
                 horizon=0.5) -> StrategicEvaluation:
        if not candidates:
            raise ValueError("at least one strategy candidate is required")
        if not 0 <= risk_sensitivity <= 1 or not 0 <= horizon <= 1:
            raise ValueError("risk_sensitivity and horizon must be in [0, 1]")
        distribution = self.engine.evaluate(candidates)
        effective_urgency = (urgency if urgency is not None
                             else max(state.soft.risk, state.soft.uncertainty))
        if not 0 <= effective_urgency <= 1:
            raise ValueError("urgency must be in [0, 1]")
        field = StrategicField(
            strategy_weights=distribution.probabilities,
            active_strategy=distribution.selected,
            confidence=distribution.confidence,
            urgency=effective_urgency,
            risk_sensitivity=risk_sensitivity,
            horizon=horizon,
            generation=state.step,
        )
        return StrategicEvaluation(distribution, field)

    def attention_bias(self, evaluation: StrategicEvaluation, token_affinity):
        return self.modulator.token_bias(evaluation.field, token_affinity)
