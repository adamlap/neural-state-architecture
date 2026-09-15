from nsa.cce import CognitiveTransactionEngine, EventKind
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState, HardState
from nsa.core.transition import TransitionProposal, TransitionValidator, state_digest


def test_soft_transition_is_atomic_and_audited():
    engine = CognitiveTransactionEngine(CanonicalState())
    tx = engine.tick(
        observation={"cpu": 0.8},
        soft_updates={"uncertainty": 0.4, "risk": 0.2},
        provenance_source="monitor",
        evidence_id="e1",
    )
    assert tx.receipt.committed
    assert engine.state.step == 1
    assert engine.state.soft.uncertainty == 0.4
    assert engine.state.provenance.sources == ("monitor",)
    assert tx.events[-1].kind == EventKind.STATE_COMMIT
    assert tx.receipt.target_digest == state_digest(engine.state)


def test_unauthorized_hard_transition_cannot_commit():
    state = CanonicalState()
    proposal = TransitionProposal(
        action_id="elevate",
        hard_transition=TransitionProposal.__dataclass_fields__["hard_transition"].default,
    )
    # A missing hard transition is harmless; explicitly test the real validator path below.
    target = HardState(authorizations=frozenset({"write"}), license_tier=2)
    bad = TransitionProposal(
        action_id="elevate",
        hard_transition=__import__("nsa.core.state", fromlist=["StateTransition"]).StateTransition(state.hard, target),
    )
    next_state, receipt = TransitionValidator().apply(state, bad)
    assert next_state == state
    assert not receipt.committed


def test_authorized_hard_transition_commits():
    state = CanonicalState()
    target = HardState(authorizations=frozenset({"write"}), license_tier=1)
    transition = __import__("nsa.core.state", fromlist=["StateTransition"]).StateTransition(state.hard, target).authorize("cap-1")
    tx = CognitiveTransactionEngine(state).tick(action_id="elevate", hard_transition=transition)
    assert tx.receipt.committed
    assert tx.state.hard == target


def test_policy_denial_prevents_execution():
    executed = []
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        policy=lambda state, action: (False, "denied by test"),
        executor=lambda action, state: executed.append(action.action_id),
    )
    tx = engine.tick(action_candidates=[ActionCandidate("danger", expected_utility=1.0, risk=0.9)])
    assert not tx.executed
    assert executed == []
    assert not tx.receipt.committed
    assert engine.state.step == 0


def test_execution_happens_after_structural_validation():
    executed = []
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        executor=lambda action, state: executed.append(action.action_id) or {"ok": True},
    )
    tx = engine.tick(
        action_candidates=[ActionCandidate("safe", expected_utility=0.5)],
        soft_updates={"unknown": 1.0},
    )
    assert not tx.executed
    assert executed == []
    assert not tx.receipt.committed
