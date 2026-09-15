from pathlib import Path

from nsa.cce.loop import CognitiveLoop
from nsa.cce.persistence import TrajectoryJournal
from nsa.cce.transaction import CognitiveTransactionEngine
from nsa.cognition.deliberation import InformationSeekingPlanner, UncertaintyDrivenDeliberator
from nsa.cognition.interfaces import ActionCandidate, Prediction
from nsa.core.capabilities import CapabilityAuthority, TrustTier
from nsa.core.state import CanonicalState


def test_capability_is_not_consumed_by_safety_rejection():
    state = CanonicalState()
    authority = CapabilityAuthority(b"test-secret")
    token = authority.mint_capability("test", "danger", "x", TrustTier.T3_SIDE_EFFECTS)
    engine = CognitiveTransactionEngine(
        state,
        capability_authority=authority,
        capability_tokens={"danger": token},
        safety_gate=lambda _s, _a: (False, "safety rejection"),
        executor=lambda *_: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    action = ActionCandidate("danger", risk=0.2, reversible=False, required_capabilities=("danger",))
    result = engine.tick(action_candidates=(action,))
    assert not result.executed
    ok, _ = authority.verify_capability(token, "danger", TrustTier.T3_SIDE_EFFECTS)
    assert ok, "rejected proposals must not burn capability nonces"


def test_information_seeking_is_explicit_and_governed():
    state = CanonicalState().observe(uncertainty=0.9)
    planner = InformationSeekingPlanner(threshold=0.65, action_factory=lambda need: ActionCandidate(
        "gather_evidence", payload={"target": need.target}, expected_utility=0.7,
        risk=0.1, reversible=True, required_capabilities=("read",)))
    action = planner.plan(state, target="unknown_fact")
    assert action is not None
    assert action.required_capabilities == ("read",)


def test_information_gain_is_not_double_counted():
    class Gain:
        def expected_gain(self, _state, _action):
            return 1.0
    state = CanonicalState().observe(uncertainty=0.9)
    candidates = (ActionCandidate("a", expected_utility=0.2),)
    decision = UncertaintyDrivenDeliberator(gain_weight=0.5).deliberate(state, candidates, information_model=Gain())
    assert decision.action is candidates[0]


def test_trajectory_journal_verifies_latest_digest(tmp_path: Path):
    engine = CognitiveTransactionEngine(CanonicalState())
    engine.tick(soft_updates={"uncertainty": 0.2}, reason="observe")
    journal = TrajectoryJournal(tmp_path / "trajectory.jsonl")
    journal.append_trajectory(engine.trajectory)
    ok, reason = journal.verify(engine.state)
    assert ok, reason


def test_prediction_error_becomes_typed_cycle_data():
    class Predictor:
        def predict(self, state, *, horizon=1):
            return Prediction(value=state, confidence=0.8, horizon=horizon)
    engine = CognitiveTransactionEngine(CanonicalState())
    loop = CognitiveLoop(engine, predictor=Predictor())
    cycle = loop.cycle(1)
    assert cycle.prediction_error is not None
    assert cycle.prediction_error.magnitude == 0.0
