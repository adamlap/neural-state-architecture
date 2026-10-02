"""Fast deterministic strategic scenario evaluation."""
from __future__ import annotations
import math
from typing import Iterable, Sequence
from .model import StrategyCandidate, StrategyDistribution

class StrategicScenarioEngine:
    """Score candidate futures without invoking an LLM.

    A learned/System-1 predictor can populate ScenarioOutcome values later
    without changing this control contract.
    """
    def __init__(self, *, risk_weight=0.30, cost_weight=0.15,
                 time_weight=0.10, information_weight=0.10,
                 reversibility_weight=0.05):
        self.risk_weight = risk_weight
        self.cost_weight = cost_weight
        self.time_weight = time_weight
        self.information_weight = information_weight
        self.reversibility_weight = reversibility_weight

    def score(self, candidate: StrategyCandidate) -> float:
        progress = sum(o.probability * o.goal_progress for o in candidate.outcomes)
        risk = sum(o.probability * o.risk for o in candidate.outcomes)
        cost = sum(o.probability * o.cost for o in candidate.outcomes)
        time = sum(o.probability * o.time for o in candidate.outcomes)
        return (progress - self.risk_weight*risk - self.cost_weight*cost
                - self.time_weight*time
                + self.information_weight*candidate.information_gain
                + self.reversibility_weight*candidate.reversibility)

    def evaluate(self, candidates: Sequence[StrategyCandidate]) -> StrategyDistribution:
        if not candidates:
            raise ValueError("at least one strategy candidate is required")
        scores = {c.strategy_id: self.score(c) for c in candidates}
        priors = {c.strategy_id: c.prior for c in candidates}
        max_score = max(scores.values())
        logits = {k: (v-max_score) + math.log(max(1e-9, priors[k] or 1.0))
                  for k, v in scores.items()}
        max_logit = max(logits.values())
        exp = {k: math.exp(v-max_logit) for k, v in logits.items()}
        total = sum(exp.values())
        probabilities = {k: v/total for k, v in exp.items()}
        selected = max(probabilities, key=probabilities.get)
        return StrategyDistribution(probabilities, selected, probabilities[selected],
                                    sum(probabilities[k]*scores[k] for k in scores))

    def evaluate_many(self, candidates: Iterable[StrategyCandidate]) -> StrategyDistribution:
        return self.evaluate(tuple(candidates))
