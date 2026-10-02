from nsa.core.state import CanonicalState, GoalState, SoftState
from nsa.strategy import (
    ScenarioOutcome, StrategyCandidate, StrategicAttentionModulator,
    StrategicController, StrategicScenarioEngine,
)

def candidate(name, progress, risk, cost=0.1, prior=0.5):
    return StrategyCandidate(
        name,
        (ScenarioOutcome("success", 0.8, progress, cost=cost, risk=risk),
         ScenarioOutcome("failure", 0.2, max(0, progress-0.4), cost=cost, risk=min(1, risk+0.1))),
        prior=prior,
    )

def test_engine_evaluates_parallel_futures():
    engine = StrategicScenarioEngine()
    result = engine.evaluate([candidate("safe", .8, .1), candidate("fast", .9, .4)])
    assert set(result.probabilities) == {"safe", "fast"}
    assert abs(sum(result.probabilities.values()) - 1) < 1e-9
    assert result.selected in result.probabilities

def test_controller_creates_bounded_field_from_canonical_state():
    state = CanonicalState(
        goals=GoalState(goals=("finish",), active_goal="finish"),
        soft=SoftState(uncertainty=.2, risk=.3, confidence=.8),
    )
    evaluation = StrategicController().evaluate(
        state, [candidate("safe", .8, .1), candidate("fast", .9, .4)]
    )
    assert evaluation.field.active_strategy == evaluation.distribution.selected
    assert evaluation.field.generation == state.step
    assert 0 <= evaluation.field.strength <= 1

def test_attention_bias_is_small_and_strategy_weighted():
    state = CanonicalState(soft=SoftState(uncertainty=.1, risk=.1, confidence=.9))
    evaluation = StrategicController().evaluate(
        state, [candidate("safe", .8, .1), candidate("fast", .9, .4)]
    )
    modulator = StrategicAttentionModulator(max_bias=.2)
    bias = modulator.token_bias(
        evaluation.field,
        [{"safe": 1.0}, {"fast": 1.0}, {"unrelated": 1.0}],
    )
    assert len(bias) == 3
    assert all(abs(v) <= .2 for v in bias)

def test_modulation_can_be_applied_to_plain_attention_logits():
    field = StrategicController().evaluate(
        CanonicalState(), [candidate("safe", .8, .1), candidate("fast", .9, .4)]
    ).field
    modulator = StrategicAttentionModulator(max_bias=.1)
    logits = (0.0, 1.0)
    result = modulator.apply(logits, field=field, token_affinity=[{"safe": 1}, {"fast": 1}])
    assert len(result) == 2
    assert result != logits
