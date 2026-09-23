"""Tests for closed-loop counterfactual simulation."""
from nsa.cognition.counterfactual import CounterfactualSimulator
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState, SoftState


def test_counterfactual_safe_action_selection():
    simulator = CounterfactualSimulator(max_acceptable_risk=0.50, risk_escalation_limit=0.30)
    current_state = CanonicalState(soft=SoftState(uncertainty=0.3, risk=0.1, confidence=0.9))

    safe_candidate_1 = ActionCandidate(
        action_id="read_system_info",
        payload={},
        expected_utility=0.8,
        risk=0.05,
    )
    safe_candidate_2 = ActionCandidate(
        action_id="optimize_cache",
        payload={},
        expected_utility=0.9,
        risk=0.08,
    )
    unsafe_candidate = ActionCandidate(
        action_id="delete_root_storage",
        payload={},
        expected_utility=0.95,
        risk=0.60,  # Exceeds max_acceptable_risk
    )

    evaluation = simulator.evaluate_candidates(
        current_state,
        [safe_candidate_1, safe_candidate_2, unsafe_candidate],
    )

    assert evaluation.evaluated_count == 3
    assert evaluation.rejection_count == 1
    # Unsafe candidate rejected
    unsafe_branch = next(b for b in evaluation.branches if b.action_id == "delete_root_storage")
    assert unsafe_branch.safe is False
    assert "exceeds threshold" in unsafe_branch.rejection_reason

    # Recommended action should be safe_candidate_2 (higher net utility: 0.9 - 0.18 > 0.8 - 0.15)
    assert evaluation.recommended_action is not None
    assert evaluation.recommended_action.action_id == "optimize_cache"
    assert evaluation.recommended_action.safe is True


def test_counterfactual_risk_escalation_gating():
    simulator = CounterfactualSimulator(max_acceptable_risk=0.80, risk_escalation_limit=0.25)
    current_state = CanonicalState(soft=SoftState(uncertainty=0.2, risk=0.1))

    escalating_action = ActionCandidate(
        action_id="escalating_operation",
        payload={},
        expected_utility=0.7,
        risk=0.35,  # Delta is 0.35 > 0.25
    )

    evaluation = simulator.evaluate_candidates(current_state, [escalating_action])
    assert evaluation.rejection_count == 1
    assert evaluation.recommended_action is None
    assert "risk escalation delta" in evaluation.branches[0].rejection_reason


def test_counterfactual_missing_capability_rejection():
    simulator = CounterfactualSimulator()
    current_state = CanonicalState()  # empty authorizations

    privileged_action = ActionCandidate(
        action_id="privileged_cmd",
        payload={},
        expected_utility=0.9,
        risk=0.1,
        required_capabilities=("admin:write",),
    )

    evaluation = simulator.evaluate_candidates(current_state, [privileged_action])
    assert evaluation.rejection_count == 1
    assert "missing required capabilities" in evaluation.branches[0].rejection_reason