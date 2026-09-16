"""Async model-to-effect orchestration for the canonical CCE boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from nsa.cce.async_transaction import AsyncCognitiveTransactionEngine
from nsa.cce.transaction import CognitiveTransaction
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.model import CognitiveContext, CognitiveModel, CognitiveProposal

AsyncEffect = Callable[[ActionCandidate, Any], Awaitable[Any]]


@dataclass(frozen=True)
class AsyncCognitiveCycleResult:
    proposal: CognitiveProposal
    transaction: CognitiveTransaction


class AsyncCognitiveOrchestrator:
    """Run model proposals through validation, async effects, then commit.

    This is intentionally separate from the synchronous orchestrator: an async
    effect must finish before canonical state claims that an action occurred.
    """

    def __init__(self, runtime: AsyncCognitiveTransactionEngine, model: CognitiveModel) -> None:
        self.runtime = runtime
        self.model = model

    async def cycle(
        self,
        context: CognitiveContext,
        *,
        observation: Any = None,
        executor: AsyncEffect | None = None,
    ) -> AsyncCognitiveCycleResult:
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
            "model_metadata": dict(proposal.metadata),
        }
        if candidates != proposal.action_candidates:
            semantic["materialized_information_actions"] = tuple(c.action_id for c in candidates)
        errors = []
        for item in proposal.prediction_errors:
            try:
                errors.append(float(item.get("magnitude", 0.0)))
            except (TypeError, ValueError):
                pass
        prediction_error = max(0.0, min(1.0, max(errors, default=0.0)))
        soft_updates = {
            "uncertainty": max(0.0, min(1.0, max(self.runtime.state.soft.uncertainty, prediction_error, 1.0 - proposal.confidence))),
            "confidence": max(0.0, min(1.0, proposal.confidence)),
        }
        transaction = await self.runtime.tick_async(
            action_candidates=tuple(candidates),
            observation=observation,
            semantic_update=semantic,
            soft_updates=soft_updates,
            action_id="model_cognitive_cycle",
            reason=proposal.rationale or "model cognitive proposal",
            provenance_source=str(proposal.metadata.get("provider", "cognitive-model")),
            evidence_id=proposal.metadata.get("evidence_id"),
            metadata={"model_proposal": True, **dict(proposal.metadata)},
            executor=executor,
        )
        return AsyncCognitiveCycleResult(proposal=proposal, transaction=transaction)

    @staticmethod
    def _information_actions(context: CognitiveContext, proposal: CognitiveProposal) -> tuple[ActionCandidate, ...]:
        if not proposal.information_needs:
            return ()
        candidates: list[ActionCandidate] = []
        for need in proposal.information_needs:
            preferred = set(need.preferred_capabilities)
            for tool in tuple(getattr(context, "tools", ()) or ()):
                if isinstance(tool, dict):
                    name, capability = tool.get("name"), tool.get("capability")
                    risk = float(tool.get("risk", 0.0)); reversible = bool(tool.get("reversible", True))
                else:
                    name, capability = getattr(tool, "name", None), getattr(tool, "capability", None)
                    risk = float(getattr(tool, "risk", 0.0)); reversible = bool(getattr(tool, "reversible", True))
                if not name or not capability or (preferred and capability not in preferred):
                    continue
                candidates.append(ActionCandidate(
                    action_id=str(name), payload={"question": need.question},
                    expected_utility=need.expected_information_gain * max(need.urgency, 0.1),
                    risk=risk, reversible=reversible, required_capabilities=(str(capability),),
                ))
                break
        return tuple(candidates)


__all__ = ["AsyncCognitiveCycleResult", "AsyncCognitiveOrchestrator", "AsyncEffect"]
