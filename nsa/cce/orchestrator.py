"""Model-to-CCE orchestration with a strict proposal/effect boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nsa.cce.canonical_runtime import CanonicalCCERuntime, TickInput
from nsa.cce.transaction import CognitiveTransaction
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
        semantic = {
            "belief_updates": tuple(proposal.belief_updates),
            "predictions": tuple(proposal.predictions),
            "prediction_errors": tuple(proposal.prediction_errors),
            "information_needs": tuple(proposal.information_needs),
            "goal_updates": tuple(proposal.goal_updates),
            "rationale": proposal.rationale,
            "model_confidence": proposal.confidence,
            "model_metadata": dict(proposal.metadata),
        }
        soft_updates = self._soft_updates(proposal)
        tick = TickInput(
            observation=observation,
            semantic_update=semantic,
            soft_updates=soft_updates,
            action_candidates=proposal.action_candidates,
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
