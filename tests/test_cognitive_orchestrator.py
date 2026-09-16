from __future__ import annotations

import asyncio

import pytest

from nsa.cce import CanonicalCCERuntime, CognitiveOrchestrator
from nsa.cognition import ActionCandidate, CognitiveContext, CognitiveProposal, InformationNeedProposal, ToolSpec
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


class InformationSeekingModel:
    async def propose(self, context):
        return CognitiveProposal(
            information_needs=(
                InformationNeedProposal(
                    "What is the current weather?",
                    expected_information_gain=0.9,
                    urgency=0.8,
                    preferred_capabilities=("weather.read",),
                ),
            ),
            rationale="uncertainty requires evidence",
            confidence=0.4,
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
    assert runtime.state.soft.confidence == pytest.approx(0.7)
    assert runtime.state.soft.uncertainty == pytest.approx(0.3)


def test_information_need_becomes_governed_action_only_from_supplied_tool():
    runtime = CanonicalCCERuntime(CanonicalState())
    context = CognitiveContext(
        state=runtime.state,
        task="epistemic_cycle",
        tools=(ToolSpec("read_weather", "weather.read", risk=0.05),),
    )
    result = asyncio.run(CognitiveOrchestrator(runtime, InformationSeekingModel()).cycle(context))

    assert result.transaction.selected_action is not None
    assert result.transaction.selected_action.action_id == "read_weather"
    assert result.transaction.selected_action.required_capabilities == ("weather.read",)
    assert result.transaction.receipt.committed is True
