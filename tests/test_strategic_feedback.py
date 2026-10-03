from nsa.core.state import CanonicalState, GoalState
from nsa.cognition.interfaces import ActionCandidate, PredictionError
from nsa.strategy import (
    StrategicController,
    StrategicAttentionModulator,
    StrategicFeedback,
    StrategicFeedbackUpdater,
    StrategicUpdatePolicy,
    SystemOneStrategyProvider,
    StrategyCandidate,
    ScenarioOutcome,
)


def test_system_one_populates_strategic_candidates():
    state = CanonicalState(goals=GoalState(goals=("deploy",), active_goal="deploy"))
    actions = [
        ActionCandidate("deploy", expected_utility=.9, risk=.1),
        ActionCandidate("danger", expected_utility=.8, risk=.8),
    ]
    decisions = SystemOneStrategyProvider().candidates(state, actions)
    assert len(decisions) == 2
    assert all(abs(sum(o.probability for o in x.candidate.outcomes) - 1) < 1e-9 for x in decisions)


def test_capability_boundary_is_preserved():
    state = CanonicalState()
    actions = [ActionCandidate("protected", required_capabilities=("admin",))]
    assert SystemOneStrategyProvider().strategic_candidates(state, actions) == ()


def test_end_to_end_strategy_to_attention_bias():
    state = CanonicalState()
    actions = [
        ActionCandidate("safe", expected_utility=.9, risk=.1),
        ActionCandidate("fast", expected_utility=.7, risk=.2),
    ]
    provider = SystemOneStrategyProvider()
    evaluation = StrategicController().evaluate(
        state, provider.strategic_candidates(state, actions)
    )
    bias = StrategicAttentionModulator(.1).token_bias(
        evaluation.field, [{"safe": 1.}, {"fast": 1.}]
    )
    assert len(bias) == 2 and all(abs(x) <= .1 for x in bias)


def test_prediction_error_updates_only_soft_prior():
    candidates = (
        StrategyCandidate(
            "safe",
            (ScenarioOutcome("ok", 1.0, goal_progress=.8, risk=.1),),
            prior=.5,
        ),
        StrategyCandidate(
            "other",
            (ScenarioOutcome("ok", 1.0, goal_progress=.5, risk=.2),),
            prior=.5,
        ),
    )
    feedback = StrategicFeedback(
        "safe",
        PredictionError(magnitude=.8),
        observed_goal_progress=.9,
        observed_risk=.1,
    )
    updated = StrategicFeedbackUpdater().update(candidates, feedback)
    assert updated[0].prior > candidates[0].prior
    assert updated[1] == candidates[1]
    assert updated[0].outcomes == candidates[0].outcomes


def test_prediction_error_without_quality_signal_does_not_guess():
    candidate = StrategyCandidate(
        "safe", (ScenarioOutcome("ok", 1.0, goal_progress=.8),), prior=.5
    )
    feedback = StrategicFeedback("safe", PredictionError(magnitude=1.0))
    assert StrategicFeedbackUpdater().update((candidate,), feedback) == (candidate,)


def test_feedback_is_bounded():
    candidate = StrategyCandidate(
        "safe", (ScenarioOutcome("ok", 1.0, goal_progress=1.0),), prior=.99
    )
    feedback = StrategicFeedback(
        "safe",
        PredictionError(magnitude=100.0),
        observed_goal_progress=1.0,
        observed_risk=0.0,
    )
    updated = StrategicFeedbackUpdater(max_delta=.05).update((candidate,), feedback)
    assert updated[0].prior <= 1.0
    assert updated[0].prior - candidate.prior <= .05


def test_update_policy_responds_to_meaningful_soft_state_changes():
    policy = StrategicUpdatePolicy(
        risk_threshold=.1, uncertainty_threshold=.1, step_interval=3
    )
    previous = CanonicalState()
    unchanged = CanonicalState(step=1)
    risk_change = CanonicalState(step=1, soft=previous.soft.__class__(
        risk=.25, uncertainty=previous.soft.uncertainty
    ))
    assert policy.should_update(previous, unchanged) is False
    assert policy.should_update(previous, risk_change) is True
    assert policy.should_update(previous, CanonicalState(step=3)) is True
