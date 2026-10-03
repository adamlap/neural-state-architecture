from nsa.core.state import CanonicalState,GoalState
from nsa.cognition.interfaces import ActionCandidate
from nsa.strategy import StrategicController,SystemOneStrategyProvider,StrategicAttentionModulator
def test_system_one_populates_strategic_candidates():
    state=CanonicalState(goals=GoalState(goals=("deploy",),active_goal="deploy"))
    actions=[ActionCandidate("deploy",expected_utility=.9,risk=.1),ActionCandidate("danger",expected_utility=.8,risk=.8)]
    decisions=SystemOneStrategyProvider().candidates(state,actions)
    assert len(decisions)==2
    assert all(abs(sum(o.probability for o in x.candidate.outcomes)-1)<1e-9 for x in decisions)
def test_capability_boundary_is_preserved():
    state=CanonicalState()
    actions=[ActionCandidate("protected",required_capabilities=("admin",))]
    assert SystemOneStrategyProvider().strategic_candidates(state,actions)==()
def test_end_to_end_strategy_to_attention_bias():
    state=CanonicalState()
    actions=[ActionCandidate("safe",expected_utility=.9,risk=.1),ActionCandidate("fast",expected_utility=.7,risk=.2)]
    provider=SystemOneStrategyProvider()
    evaluation=StrategicController().evaluate(state,provider.strategic_candidates(state,actions))
    bias=StrategicAttentionModulator(.1).token_bias(evaluation.field,[{"safe":1.},{"fast":1.}])
    assert len(bias)==2 and all(abs(x)<=.1 for x in bias)