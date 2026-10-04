"""Bridge NSA System One decisions into strategic future candidates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.system_one import CalibratedDecision,SystemOneDecisionEngine
from nsa.core.state import CanonicalState
from .model import ScenarioOutcome,StrategyCandidate
@dataclass(frozen=True)
class StrategyDecision:
    candidate:StrategyCandidate
    decision:CalibratedDecision
class SystemOneStrategyProvider:
    """Uses existing SystemOneDecisionEngine as the strategic scenario prior.

    This is an adapter, not a replacement for System One or CCE. Capability
    checks remain mandatory before a candidate can enter the strategic field.
    """
    def __init__(self,engine=None):
        self.engine=engine or SystemOneDecisionEngine()
    def candidates(self,state:CanonicalState,actions:Sequence[ActionCandidate])->tuple[StrategyDecision,...]:
        result=[]
        for action in actions:
            if any(not state.hard.has_permission(c) for c in action.required_capabilities): continue
            d=self.engine.evaluate_candidate(state,action)
            execute=d.probabilities.get("execute",0.)
            utility=max(0.,min(1.,action.expected_utility))
            success=ScenarioOutcome("execute",execute,utility,cost=max(0.,1.-utility)*.25,risk=action.risk)
            failure=ScenarioOutcome("reject",1.-execute,0.,cost=0.,risk=0.)
            result.append(StrategyDecision(StrategyCandidate(action.action_id,(success,failure),prior=max(execute,1e-6),reversibility=1. if action.reversible else .1),d))
        return tuple(result)
    def strategic_candidates(self,state,actions):
        return tuple(x.candidate for x in self.candidates(state,actions))