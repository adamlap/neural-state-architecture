"""Hardening tests for the canonical CCE boundary."""
from __future__ import annotations

import importlib

from nsa.cce.effects import EffectReceipt, TwoPhaseExecutor
from nsa.cce.persistence import TrajectoryJournal
from nsa.cce.transaction import CognitiveTransactionEngine
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, TrustTier
from nsa.core.capability_constraints import CapabilityConstraintEvaluator, ConstraintContext
from nsa.core.state import CanonicalState
from nsa.core.state_codec import decode_state, encode_state


class FakeEffect:
    def __init__(self):
        self.prepared = 0
        self.committed = 0
        self.aborted = 0
        self.compensated = 0

    def prepare(self, action, state):
        self.prepared += 1
        return {"action": action.action_id}

    def commit(self, prepared):
        self.committed += 1
        return prepared

    def abort(self, prepared):
        self.aborted += 1

    def compensate(self, result):
        self.compensated += 1


def test_state_codec_round_trip_preserves_digest():
    state = CanonicalState().observe(uncertainty=0.4, confidence=0.8)
    assert decode_state(encode_state(state)) == state


def test_journal_reconstructs_complete_state(tmp_path):
    state = CanonicalState().with_semantic({"goal": "test"})
    journal = TrajectoryJournal(tmp_path / "trajectory.jsonl")
    from nsa.cce.trajectory import CognitiveTrajectory
    trajectory = CognitiveTrajectory(state)
    journal.append(trajectory.latest, state=state)
    assert journal.verify(state)[0]
    assert journal.replay_states() == (state,)


def test_constraint_target_and_rate_limit():
    authority = CapabilityAuthority(b"test-secret")
    token = authority.mint_capability(
        "test", "inspect", "docs", TrustTier.T1_INFO_GATHER,
        constraints={"targets": ["doc-1"], "max_calls": 1, "window_seconds": 60},
    )
    action = ActionCandidate("inspect", expected_utility=1.0, required_capabilities=("docs",))
    evaluator = CapabilityConstraintEvaluator()
    context = ConstraintContext(target="doc-1", scope="docs", now=10.0)
    assert evaluator.evaluate(token, action, context=context).allowed
    evaluator.record_use(token, action, context=context)
    assert not evaluator.evaluate(token, action, context=context).allowed


def test_rejected_policy_does_not_consume_capability(tmp_path):
    authority = CapabilityAuthority(b"test-secret")
    token = authority.mint_capability("test", "danger", "danger", TrustTier.T3_SIDE_EFFECTS)
    action = ActionCandidate("danger", risk=0.4, reversible=False, required_capabilities=("danger",))
    engine = CognitiveTransactionEngine(
        CanonicalState(), policy=lambda state, action: (False, "blocked"),
        capability_authority=authority, capability_tokens={"danger": token},
    )
    result = engine.tick(action_candidates=(action,))
    assert not result.receipt.committed
    assert authority.verify_capability(token, "danger", TrustTier.T3_SIDE_EFFECTS)[0]


def test_two_phase_effect_commits_before_state_receipt():
    effect = FakeEffect()
    action = ActionCandidate("effect", expected_utility=1.0)
    engine = CognitiveTransactionEngine(CanonicalState(), effect_executor=TwoPhaseExecutor(effect))
    result = engine.tick(action_candidates=(action,), semantic_update={"committed": True})
    assert result.executed
    assert result.effect_receipt is not None and result.effect_receipt.committed
    assert effect.prepared == 1 and effect.committed == 1
    assert result.state.semantic.value == {"committed": True}


def test_canonical_cce_package_does_not_eagerly_import_neural_substrate():
    importlib.import_module("nsa.cce")
    assert "nsa.cce.omega_adapter" not in __import__("sys").modules
