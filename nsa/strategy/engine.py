from __future__ import annotations
import math
from typing import Sequence
from .model import StrategyCandidate,StrategyDistribution
class StrategicScenarioEngine:
    """Cheap deterministic evaluator for competing futures."""
    def __init__(self,*,risk_weight=.30,cost_weight=.15,time_weight=.10,information_weight=.10,reversibility_weight=.05):
        self.risk_weight=risk_weight; self.cost_weight=cost_weight; self.time_weight=time_weight
        self.information_weight=information_weight; self.reversibility_weight=reversibility_weight
    def score(self,c:StrategyCandidate)->float:
        p=sum(o.probability*o.goal_progress for o in c.outcomes)
        r=sum(o.probability*o.risk for o in c.outcomes)
        cost=sum(o.probability*o.cost for o in c.outcomes)
        t=sum(o.probability*o.time for o in c.outcomes)
        return p-self.risk_weight*r-self.cost_weight*cost-self.time_weight*t+self.information_weight*c.information_gain+self.reversibility_weight*c.reversibility
    def evaluate(self,candidates:Sequence[StrategyCandidate])->StrategyDistribution:
        if not candidates: raise ValueError("at least one strategy candidate is required")
        scores={c.strategy_id:self.score(c) for c in candidates}
        logits={c.strategy_id:scores[c.strategy_id]+math.log(max(1e-9,c.prior or 1.)) for c in candidates}
        m=max(logits.values()); ex={k:math.exp(v-m) for k,v in logits.items()}; total=sum(ex.values())
        probs={k:v/total for k,v in ex.items()}; selected=max(probs,key=probs.get)
        return StrategyDistribution(probs,selected,probs[selected],sum(probs[k]*scores[k] for k in scores))