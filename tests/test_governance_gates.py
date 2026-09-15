from nsa.cce import CognitiveTransactionEngine
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, TrustTier
from nsa.core.state import CanonicalState


def test_missing_capability_blocks_execution_and_is_trajectoried():
    executed = []
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        executor=lambda action, state: executed.append(action.action_id),
    )
    action = ActionCandidate(
        "write_file",
        expected_utility=1.0,
        risk=0.2,
        required_capabilities=("filesystem.write",),
    )
    tx = engine.tick(action_candidates=[action])
    assert not tx.executed
    assert executed == []
    assert not tx.receipt.committed
    assert engine.trajectory.latest.receipt == tx.receipt


def test_capability_token_authorizes_reversible_action():
    authority = CapabilityAuthority(master_secret_key=b"test-secret")
    token = authority.mint_capability(
        principal="test",
        action_id="read_remote",
        scope="remote",
        target_tier=TrustTier.T2_REVERSIBLE,
    )
    executed = []
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        executor=lambda action, state: executed.append(action.action_id) or "ok",
        capability_authority=authority,
        capability_tokens={"network.read": token},
    )
    action = ActionCandidate(
        "read_remote",
        expected_utility=1.0,
        risk=0.1,
        reversible=True,
        required_capabilities=("network.read",),
    )
    tx = engine.tick(action_candidates=[action])
    assert tx.executed
    assert executed == ["read_remote"]
    assert tx.receipt.committed


def test_capability_constraints_are_checked_before_consumption():
    authority = CapabilityAuthority(master_secret_key=b"test-secret")
    token = authority.mint_capability(
        principal="test",
        action_id="risky_read",
        scope="remote",
        target_tier=TrustTier.T2_REVERSIBLE,
        constraints={"max_risk": 0.1},
    )
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        capability_authority=authority,
        capability_tokens={"network.read": token},
    )
    action = ActionCandidate(
        "risky_read",
        expected_utility=1.0,
        risk=0.8,
        reversible=True,
        required_capabilities=("network.read",),
    )
    first = engine.tick(action_candidates=[action])
    second = engine.tick(action_candidates=[action])
    assert not first.receipt.committed
    assert not second.receipt.committed
    assert "constraint" in first.receipt.reason
    assert "constraint" in second.receipt.reason


def test_safety_gate_blocks_effect_before_execution():
    executed = []
    engine = CognitiveTransactionEngine(
        CanonicalState(),
        safety_gate=lambda state, action: (False, "ISK rejected action"),
        executor=lambda action, state: executed.append(action.action_id),
    )
    action = ActionCandidate("external_side_effect", expected_utility=1.0, risk=0.3)
    tx = engine.tick(action_candidates=[action])
    assert not tx.executed
    assert executed == []
    assert tx.receipt.reason == "ISK rejected action"
    assert engine.trajectory.latest.receipt == tx.receipt
