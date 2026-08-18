"""Tests for the explicit self-state research primitive."""

import pytest

from nsa.self_state import SelfState, SelfStateObservation


def test_self_state_is_bounded():
    state = SelfState(confidence=0.8, uncertainty=0.2, perceived_risk=0.4)
    assert state.metacognitive_pressure() == 0.4


def test_self_state_rejects_invalid_values():
    with pytest.raises(ValueError):
        SelfState(confidence=1.1)


def test_observation_updates_only_self_state():
    state = SelfState(confidence=0.9)
    observed = SelfStateObservation(
        confidence=0.6,
        uncertainty=0.7,
        state_prediction_error=0.3,
    ).apply(state)

    assert observed.confidence == 0.6
    assert observed.uncertainty == 0.7
    assert observed.state_prediction_error == 0.3
    assert observed.step == 1
    assert state.confidence == 0.9


def test_unknown_observation_field_is_rejected():
    state = SelfState()
    with pytest.raises(ValueError):
        state.observe(unknown=0.5)


def test_metacognitive_pressure_tracks_highest_signal():
    state = SelfState(
        uncertainty=0.2,
        perceived_risk=0.9,
        resource_pressure=0.4,
        state_prediction_error=0.6,
    )
    assert state.metacognitive_pressure() == 0.9
