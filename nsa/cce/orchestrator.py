"""Model-to-CCE orchestration with a strict proposal/effect boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nsa.cce.canonical_runtime import CanonicalCCERuntime, TickInput
from nsa.cce.transaction import CognitiveTransaction
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.model import CognitiveContext, CognitiveModel, CognitiveProposal


@dataclass(frozen=True)
class CognitiveCycleResult:
    """Auditable result containing both the model proposal and governed commit."""

    proposal: CognitiveProposal
    transaction: CognitiveTransaction


class CognitiveOrchestrator:
    """Run one model proposal through the canonical CCE transaction boundary.

    The model is never given an execution callback. Its output is converted to
    ordinary CCE candidates and semantic/soft proposals; policy, capability,
    safety, effects and canonical state remain inside CCE.
    """

    def __init__(self, runtime: CanonicalCCERuntime, model: CognitiveModel) -> None:
        self.runtime = runtime
        self.model = model

    async def cycle(self, context: CognitiveContext, *, observation: Any = None) -> CognitiveCycleResult:
        proposal = await self.model.propose(context)
        candidates = proposal.action_candidates or self._information_actions(context, proposal)
        semantic = {
            "belief_updates": tuple(proposal.belief_updates),
            "predictions": tuple(proposal.predictions),
            "prediction_errors": tuple(proposal.prediction_errors),
            "information_needs": tuple(proposal.information_needs),
            "goal_updates": tuple(proposal.goal_updates),
            "rationale": proposal.rationale,
            "model_confidence": proposal.confidence,
        }
        if candidates != proposal.action_candidates:
            semantic["materialized_information_actions"] = tuple(c.action_id for c in candidates)
        semantic["model_metadata"] = dict(proposal.metadata)
        soft_updates = self._soft_updates(proposal)
        tick = TickInput(
            observation=observation,
            semantic_update=semantic,
            soft_updates=soft_updates,
            action_candidates=tuple(candidates),
            action_id="model_cognitive_cycle",
            reason=proposal.rationale or "model cognitive proposal",
            provenance_source=str(proposal.metadata.get("provider", "cognitive-model")),
            evidence_id=proposal.metadata.get("evidence_id"),
            metadata={"model_proposal": True, **dict(proposal.metadata)},
        )
        transaction = self.runtime.tick(tick)
        if transaction is None:
            raise RuntimeError("CCE returned no transaction")
        return CognitiveCycleResult(proposal=proposal, transaction=transaction)

    @staticmethod
    def _information_actions(context: CognitiveContext, proposal: CognitiveProposal) -> tuple[ActionCandidate, ...]:
        """Convert explicit information needs into tool proposals, never authority.

        A model can ask a question, but it cannot invent an executor. Only tools
        supplied by the application context are eligible, and the resulting
        candidate still passes the normal CCE policy/capability/safety gates.
        """
        if not proposal.information_needs:
            return ()
        tools = tuple(getattr(context, "tools", ()) or ())
        candidates: list[ActionCandidate] = []
        for need in proposal.information_needs:
            preferred = set(need.preferred_capabilities)
            for tool in tools:
                name = getattr(tool, "name", None)
                capability = getattr(tool, "capability", None)
                if isinstance(tool, dict):
                    name = tool.get("name")
                    capability = tool.get("capability")
                if not name or not capability or (preferred and capability not in preferred):
                    continue
                risk = float(getattr(tool, "risk", tool.get("risk", 0.0) if isinstance(tool, dict) else 0.0))
                reversible = bool(getattr(tool, "reversible", tool.get("reversible", True) if isinstance(tool, dict) else True))
                candidates.append(
                    ActionCandidate(
                        action_id=str(name),
                        payload={"question": need.question},
                        expected_utility=need.expected_information_gain * max(need.urgency, 0.1),
                        risk=risk,
                        reversible=reversible,
                        required_capabilities=(str(capability),),
                    )
                )
                break
        return tuple(candidates)

    def _soft_updates(self, proposal: CognitiveProposal) -> dict[str, float]:
        state = self.runtime.state
        error_values = []
        for item in proposal.prediction_errors:
            try:
                error_values.append(float(item.get("magnitude", 0.0)))
            except (TypeError, ValueError):
                continue
        prediction_error = max(0.0, min(1.0, max(error_values, default=0.0)))
        confidence_uncertainty = 1.0 - proposal.confidence
        uncertainty = max(state.soft.uncertainty, prediction_error, confidence_uncertainty)
        return {
            "uncertainty": max(0.0, min(1.0, uncertainty)),
            "confidence": max(0.0, min(1.0, proposal.confidence)),
        }


__all__ = ["CognitiveCycleResult", "CognitiveOrchestrator"]
