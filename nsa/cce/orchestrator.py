"""Model-to-CCE orchestration with a strict proposal/effect boundary."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from nsa.cce.canonical_runtime import CanonicalCCERuntime, TickInput
from nsa.cce.transaction import CognitiveTransaction
from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.model import CognitiveContext, CognitiveModel, CognitiveProposal

@dataclass(frozen=True)
class CognitiveCycleResult:
    proposal:CognitiveProposal
    transaction:CognitiveTransaction

class CognitiveOrchestrator:
    """Run model cognition while making the causal epistemic chain explicit."""
    def __init__(self,runtime:CanonicalCCERuntime,model:CognitiveModel)->None:
        self.runtime=runtime;self.model=model

    async def cycle(self,context:CognitiveContext,*,observation:Any=None)->CognitiveCycleResult:
        proposal=await self.model.propose(context)
        step=self.runtime.state.step
        events=[
            CognitiveEvent(EventKind.COGNITION,step,f"model-{step}",{
                "task":context.task,"confidence":proposal.confidence,"rationale":proposal.rationale,
                "metadata":dict(proposal.metadata),
            },source="cognitive-model")
        ]
        for index,need in enumerate(proposal.information_needs):
            events.append(CognitiveEvent(EventKind.INFORMATION_NEED,step,f"information-need-{step}-{index}",{
                "question":need.question,"expected_information_gain":need.expected_information_gain,
                "urgency":need.urgency,"preferred_capabilities":tuple(need.preferred_capabilities),
            },source="cognitive-model"))
        for index,belief in enumerate(proposal.belief_updates):
            events.append(CognitiveEvent(EventKind.BELIEF,step,f"belief-proposal-{step}-{index}",{"belief":dict(belief)},source="cognitive-model"))
        for index,prediction in enumerate(proposal.predictions):
            events.append(CognitiveEvent(EventKind.PREDICTION,step,f"prediction-proposal-{step}-{index}",{
                "confidence":prediction.confidence,"horizon":prediction.horizon,"value":prediction.value},source="cognitive-model"))
        for index,error in enumerate(proposal.prediction_errors):
            events.append(CognitiveEvent(EventKind.PREDICTION_ERROR,step,f"prediction-error-{step}-{index}",dict(error),source="cognitive-model"))

        candidates=proposal.action_candidates or self._information_actions(context,proposal)
        if candidates:
            selected=max(candidates,key=lambda c:(c.expected_utility-c.risk,c.reversible))
            events.append(CognitiveEvent(EventKind.DELIBERATION,step,f"deliberation-{step}",{
                "selected_action":selected.action_id,"candidate_count":len(candidates),
                "expected_utility":selected.expected_utility,"risk":selected.risk,
            },source="cognitive-model"))
        semantic={"belief_updates":tuple(proposal.belief_updates),"predictions":tuple(proposal.predictions),
                  "prediction_errors":tuple(proposal.prediction_errors),"information_needs":tuple(proposal.information_needs),
                  "goal_updates":tuple(proposal.goal_updates),"rationale":proposal.rationale,
                  "model_confidence":proposal.confidence,"model_metadata":dict(proposal.metadata)}
        if candidates!=proposal.action_candidates:semantic["materialized_information_actions"]=tuple(c.action_id for c in candidates)
        soft_updates=self._soft_updates(proposal)
        tick=TickInput(observation=observation,semantic_update=semantic,soft_updates=soft_updates,
                       action_candidates=tuple(candidates),action_id="model_cognitive_cycle",
                       reason=proposal.rationale or "model cognitive proposal",
                       provenance_source=str(proposal.metadata.get("provider","cognitive-model")),
                       evidence_id=proposal.metadata.get("evidence_id"),
                       metadata={"model_proposal":True,**dict(proposal.metadata)})
        transaction=self.runtime.tick(tick)
        if transaction is None:raise RuntimeError("CCE returned no transaction")
        return CognitiveCycleResult(proposal=proposal,transaction=transaction)

    @staticmethod
    def _information_actions(context:CognitiveContext,proposal:CognitiveProposal)->tuple[ActionCandidate,...]:
        if not proposal.information_needs:return ()
        tools=tuple(getattr(context,"tools",()) or ())
        candidates=[]
        for need in proposal.information_needs:
            preferred=set(need.preferred_capabilities)
            for tool in tools:
                name=getattr(tool,"name",None);capability=getattr(tool,"capability",None)
                if isinstance(tool,dict):name=tool.get("name");capability=tool.get("capability")
                if not name or not capability or (preferred and capability not in preferred):continue
                risk=float(getattr(tool,"risk",tool.get("risk",0.0) if isinstance(tool,dict) else 0.0))
                reversible=bool(getattr(tool,"reversible",tool.get("reversible",True) if isinstance(tool,dict) else True))
                candidates.append(ActionCandidate(action_id=str(name),payload={"question":need.question},
                    expected_utility=need.expected_information_gain*max(need.urgency,.1),risk=risk,reversible=reversible,
                    required_capabilities=(str(capability),)))
                break
        return tuple(candidates)

    def _soft_updates(self,proposal:CognitiveProposal)->dict[str,float]:
        state=self.runtime.state;errors=[]
        for item in proposal.prediction_errors:
            try:errors.append(float(item.get("magnitude",0.0)))
            except (TypeError,ValueError):pass
        prediction_error=max(0.0,min(1.0,max(errors,default=0.0)))
        uncertainty=max(state.soft.uncertainty,prediction_error,1.0-proposal.confidence)
        return {"uncertainty":max(0.0,min(1.0,uncertainty)),"confidence":max(0.0,min(1.0,proposal.confidence))}

__all__=["CognitiveCycleResult","CognitiveOrchestrator"]
