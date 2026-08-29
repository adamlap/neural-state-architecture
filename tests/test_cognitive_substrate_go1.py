from nsa.cognition.substrate import CognitiveState, CognitiveSubstrate, CognitiveSwitches


def test_workspace_competition_is_deterministic_and_capacity_bounded():
    substrate = CognitiveSubstrate(workspace_capacity=1)
    state = CognitiveState()
    a = substrate.transition(state, "alpha beta", confidence=0.8)
    b = substrate.transition(state, "alpha beta", confidence=0.8)
    assert a == b
    assert len(a.workspace.active) <= 1
    assert a.workspace.ignition_count == 1


def test_prediction_error_decreases_after_repeated_observation():
    substrate = CognitiveSubstrate()
    state = CognitiveState()
    state = substrate.transition(state, 10.0)
    first = state.prediction.mean_error
    state = substrate.transition(state, 10.0)
    assert state.prediction.mean_error <= first
    assert state.prediction.update_count == 2


def test_integration_and_recurrence_are_observable():
    substrate = CognitiveSubstrate()
    state = substrate.transition(CognitiveState(), "observation", confidence=0.9)
    assert set(state.integration.nodes) == {"perception", "prediction", "attention", "workspace", "self"}
    assert 0.0 <= state.integration.integration <= 1.0
    assert 0.0 <= state.metrics.recurrence <= 1.0
    assert state.metrics.cognitive_continuity >= 0.0


def test_self_model_tracks_internal_state_and_metacognition():
    substrate = CognitiveSubstrate()
    state = substrate.transition(CognitiveState(), 1.0)
    assert state.self_model.update_count == 1
    assert state.self_model.internal_state
    assert 0.0 <= state.self_model.confidence <= 1.0
    assert 0.0 <= state.self_model.metacognitive_signal <= 1.0


def test_ablation_switches_disable_mechanisms_without_changing_runtime_shape():
    switches = CognitiveSwitches(workspace=False, recurrence=False, predictive_processing=False, self_model=False, integration=False)
    substrate = CognitiveSubstrate(switches=switches)
    state = substrate.transition(CognitiveState(switches=switches), 1.0)
    assert state.workspace.active == ()
    assert state.prediction.update_count == 0
    assert state.integration.integration == 0.0
    assert state.self_model.update_count == 0
    assert state.metrics.recurrence == 0.0


def test_information_gain_is_bounded_and_state_dependent():
    substrate = CognitiveSubstrate()
    assert substrate.information_gain(0.0, 1.0) == 0.0
    assert substrate.information_gain(1.0, 1.0) == 1.0
    assert substrate.information_gain(0.8, 0.5) == 0.4
