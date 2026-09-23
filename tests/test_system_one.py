"""Tests for System One fast typed decision engine (Jev-inspired)."""
import pytest
from nsa.cognition.system_one import SystemOneDecisionEngine, TypedDecisionSchema, CalibratedDecision
from nsa.cognition.interfaces import ActionCandidate
from nsa.cognition.counterfactual import CounterfactualSimulator
from nsa.core.state import CanonicalState, HardState, GoalState, SoftState


def test_system_one_schema_evaluation():
    engine = SystemOneDecisionEngine()
    schema = TypedDecisionSchema(
        name="routing_intent",
        choices=("rag_search", "sql_query", "direct_answer"),
        min_confidence=0.60,
        risk_weights={"rag_search": 0.1, "sql_query": 0.3, "direct_answer": 0.05},
    )

    features = {"rag_search": 3.5, "sql_query": 0.5, "direct_answer": 1.0}
    decision = engine.evaluate_schema(schema, features)

    assert decision.selected_choice == "rag_search"
    assert decision.calibrated_confidence > 0.60
    assert decision.passed_gate is True
    assert decision.risk_score == 0.1
    assert decision.latency_ms >= 0.0


def test_system_one_rlcd_calibration():
    engine = SystemOneDecisionEngine(calibration_slope=1.2, calibration_intercept=-0.1)

    low_p = engine.calibrate_probability(0.2)
    mid_p = engine.calibrate_probability(0.5)
    high_p = engine.calibrate_probability(0.9)

    assert 0.0 <= low_p < mid_p < high_p <= 1.0


def test_system_one_epistemic_uncertainty_entropy():
    engine = SystemOneDecisionEngine()

    # Maximum entropy for uniform distribution over 3 choices
    uniform = [1/3, 1/3, 1/3]
    h_max = engine.compute_epistemic_uncertainty(uniform)
    assert abs(h_max - 1.0) < 1e-4

    # Low entropy for concentrated distribution
    sharp = [0.98, 0.01, 0.01]
    h_min = engine.compute_epistemic_uncertainty(sharp)
    assert h_min < 0.20


def test_system_one_fast_candidate_pruning():
    engine = SystemOneDecisionEngine()
    state = CanonicalState().with_goal(GoalState(goals=("retrieve documents",)))

    cand1 = ActionCandidate(action_id="retrieve documents", expected_utility=0.9, risk=0.1)
    cand2 = ActionCandidate(action_id="unrelated dangerous action", expected_utility=0.1, risk=0.85)
    cand3 = ActionCandidate(action_id="admin delete database", expected_utility=0.9, risk=0.2, required_capabilities=("admin",))

    pruned = engine.fast_prune_candidates(state, [cand1, cand2, cand3], max_acceptable_risk=0.70)

    # cand2 pruned for excessive risk (> 0.70)
    # cand3 pruned for missing capability ("admin")
    # cand1 kept
    assert len(pruned) == 1
    assert pruned[0].action_id == "retrieve documents"


def test_system_one_counterfactual_integration():
    engine = SystemOneDecisionEngine()
    simulator = CounterfactualSimulator(max_acceptable_risk=0.60, system_one=engine)

    state = CanonicalState()
    candidates = [
        ActionCandidate("safe_action_1", expected_utility=0.8, risk=0.1),
        ActionCandidate("safe_action_2", expected_utility=0.95, risk=0.15),
        ActionCandidate("risky_action", expected_utility=0.99, risk=0.85),
    ]

    eval_result = simulator.evaluate_candidates(state, candidates)

    assert eval_result.recommended_action is not None
    assert eval_result.recommended_action.action_id == "safe_action_2"
    assert eval_result.rejection_count >= 1
    assert eval_result.duration_ms < 100.0  # Fast sub-100ms execution


def test_system_one_soft_state_update():
    engine = SystemOneDecisionEngine()
    state = CanonicalState()

    schema = TypedDecisionSchema(name="test", choices=("a", "b"), risk_weights={"a": 0.15, "b": 0.4})
    decision = engine.evaluate_schema(schema, {"a": 2.0, "b": 0.5})

    new_soft = engine.update_soft_state(state, decision)
    assert 0.0 <= new_soft.uncertainty <= 1.0
    assert new_soft.risk == 0.15
    assert new_soft.confidence == decision.calibrated_confidence
