import pytest

from nsa.self_state import SelfState, SelfStatePredictor


def test_prediction_error_is_zero_for_matching_state():
    state = SelfState(confidence=0.8, uncertainty=0.2)
    prediction = SelfStatePredictor().predict(state)
    assert prediction.error(state) == 0.0


def test_prediction_error_is_nonzero_after_observed_change():
    state = SelfState(confidence=0.8, uncertainty=0.2)
    prediction = SelfStatePredictor().predict(state)
    observed = state.observe(confidence=0.3, uncertainty=0.7)
    assert prediction.error(observed) > 0.0


def test_error_requires_observation():
    state = SelfState()
    with pytest.raises(ValueError):
        SelfStatePredictor().predict(state).error()
