from __future__ import annotations

import asyncio

from nsa.cce import CanonicalCCERuntime, CognitiveOrchestrator
from nsa.cognition import ActionCandidate, CognitiveContext, CognitiveProposal
from nsa.core.state import CanonicalState


class FakeModel:
    async def propose(self, context):
        return CognitiveProposal(
            belief_updates=({"fact": "observed"},),
            action_candidates=(
                ActionCandidate("observe_weather", expected_utility=0.8, risk=0.1)
            ),
            rationale="reduce uncertainty with an observation",
            confidence=0.7,
            metadata={"provider": "fake", "background": True},
        )


def test_model_is_only_a_proposal_source_and_cce_commits():
    runtime = CanonicalCCERuntime(CanonicalState())
    result = asyncio.run(
        CognitiveOrchestrator(runtime, FakeModel()).cycle(
            CognitiveContext(state=runtime.state, task="epistemic_cycle")
        )
    )

    assert result.proposal.action_candidates[0].action_id == "observe_weather"
    assert result.transaction.executed is False
    assert result.transaction.receipt.committed is True
    assert runtime.state.step == 1
    assert runtime.state.semantic.value["belief_updates"][0]["fact"] == "observed"
    assert runtime.state.soft.confidence == 0.7
    assert runtime.state.soft.uncertainty == 0.30000000000000004
