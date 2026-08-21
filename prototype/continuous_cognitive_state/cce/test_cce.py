"""R5 CCE integration tests.

These tests exercise the actual NSA ProductLattice implementation. They do not
stub or replace the governing algebra.
"""
from __future__ import annotations

import asyncio

from .action import ActionProposal
from .engine import CCEConfig, ContinuousCognitiveEngine
from .governor import CCEGovernor, CCEPolicy
from .state import CCEState


class NoopReasoner:
    def reason(self, state, event):
        return f"tick={state.tick_count};dynamic={state.dynamic.vector()};event={event!r}"


class DynamicProposal:
    def propose(self, state, reasoning):
        return ActionProposal(
            capability="deployment.capability",
            payload={"reasoning": reasoning},
            confidence=0.95,
            risk=0.1,
            provenance="live-reasoner",
            reversible=True,
        )


class RecordingActuator:
    def __init__(self):
        self.executed = []

    async def execute(self, proposal):
        self.executed.append(proposal)


def test_governance_is_runtime_policy_driven():
    proposal = ActionProposal("deployment.capability", confidence=0.95, risk=0.1)
    governor = CCEGovernor(CCEPolicy(capabilities=frozenset({"deployment.capability"})))
    source = governor.source_state(CCEState())
    target = governor.target_state(proposal)
    assert governor.evaluate(proposal, source, target).status == "ALLOW"

    denied = CCEGovernor(CCEPolicy(capabilities=frozenset()))
    assert denied.evaluate(proposal, denied.source_state(CCEState()), denied.target_state(proposal)).status == "DENY"


def test_state_evolves_without_external_input():
    state = CCEState()
    before = state.dynamic.vector()
    engine = ContinuousCognitiveEngine(
        state=state,
        reasoner=NoopReasoner(),
        governor=CCEGovernor(CCEPolicy()),
        proposal_generator=DynamicProposal(),
        config=CCEConfig(state_dt=0.1),
    )
    asyncio.run(engine.tick())
    after = state.dynamic.vector()
    assert before != after


def test_allowed_proposal_reaches_only_real_actuator():
    actuator = RecordingActuator()
    engine = ContinuousCognitiveEngine(
        state=CCEState(),
        reasoner=NoopReasoner(),
        governor=CCEGovernor(CCEPolicy(capabilities=frozenset({"deployment.capability"}))),
        proposal_generator=DynamicProposal(),
        actuator=actuator,
    )
    decision = asyncio.run(engine.tick())
    assert decision is not None
    assert decision.status == "ALLOW"
    assert len(actuator.executed) == 1
