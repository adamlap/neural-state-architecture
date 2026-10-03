from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Tuple
def _b(name,v):
    if not 0.0<=v<=1.0: raise ValueError(f"{name} must be in [0, 1]")
@dataclass(frozen=True)
class ScenarioOutcome:
    outcome_id:str; probability:float; goal_progress:float; cost:float=0.; risk:float=0.; time:float=0.
    def __post_init__(self):
        for n in ("probability","goal_progress","cost","risk","time"): _b(n,getattr(self,n))
@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id:str; outcomes:Tuple[ScenarioOutcome,...]; prior:float=.5; information_gain:float=0.; reversibility:float=1.
    def __post_init__(self):
        if not self.strategy_id or not self.outcomes: raise ValueError("strategy_id and outcomes are required")
        for n in ("prior","information_gain","reversibility"): _b(n,getattr(self,n))
        if abs(sum(x.probability for x in self.outcomes)-1.)>1e-6: raise ValueError("outcome probabilities must sum to 1")
@dataclass(frozen=True)
class StrategyDistribution:
    probabilities:Mapping[str,float]; selected:str; confidence:float; expected_value:float
@dataclass(frozen=True)
class StrategicField:
    strategy_weights:Mapping[str,float]=field(default_factory=dict)
    active_strategy:str|None=None; confidence:float=0.; urgency:float=0.; risk_sensitivity:float=.5; horizon:float=.5; generation:int=0
    def __post_init__(self):
        for n in ("confidence","urgency","risk_sensitivity","horizon"): _b(n,getattr(self,n))
        for v in self.strategy_weights.values(): _b("strategy weight",v)
    @property
    def strength(self): return min(1.,self.confidence*(.5+.5*self.urgency))