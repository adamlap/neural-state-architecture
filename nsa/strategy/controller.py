from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from nsa.core.state import CanonicalState
from .engine import StrategicScenarioEngine
from .model import StrategyCandidate,StrategyDistribution,StrategicField
from .modulation import StrategicAttentionModulator
@dataclass(frozen=True)
class StrategicEvaluation:
    distribution:StrategyDistribution; field:StrategicField
class StrategicController:
    def __init__(self,engine=None,modulator=None):
        self.engine=engine or StrategicScenarioEngine(); self.modulator=modulator or StrategicAttentionModulator()
    def evaluate(self,state:CanonicalState,candidates:Sequence[StrategyCandidate],*,urgency=None,risk_sensitivity=.5,horizon=.5):
        d=self.engine.evaluate(candidates)
        u=max(state.soft.risk,state.soft.uncertainty) if urgency is None else urgency
        if not 0<=u<=1: raise ValueError("urgency must be in [0, 1]")
        f=StrategicField(d.probabilities,d.selected,d.confidence,u,risk_sensitivity,horizon,state.step)
        return StrategicEvaluation(d,f)
    def attention_bias(self,evaluation,affinity): return self.modulator.token_bias(evaluation.field,affinity)