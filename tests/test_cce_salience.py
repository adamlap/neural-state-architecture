import pytest

from nsa.runtime.cce_salience import (
    AdaptiveSalienceGate,
    SalienceObservation,
)


def test_quiet_observation_does_not_trigger_initially():
    gate = AdaptiveSalienceGate()
    decision = gate.observe(SalienceObservation())
    assert decision.score == 0.0
    assert decision.triggered is False


def test_large_prediction_error_triggers_cognitive_event():
    gate = AdaptiveSalienceGate()
    decision = gate.observe(SalienceObservation(prediction_error=1.0))
    assert decision.score > 0.0
    assert decision.triggered is True


def test_baseline_adapts_to_repeated_activity():
    gate = AdaptiveSalienceGate(baseline_decay=0.5)
    first = gate.observe(SalienceObservation(input_delta=1.0))
    second = gate.observe(SalienceObservation(input_delta=1.0))
    assert second.baseline > first.baseline
    assert second.threshold > first.threshold


def test_non_finite_or_negative_signal_fails_closed():
    gate = AdaptiveSalienceGate()
    with pytest.raises(ValueError):
        gate.observe(SalienceObservation(prediction_error=float("nan")))
    with pytest.raises(ValueError):
        gate.observe(SalienceObservation(state_delta=-1.0))


def test_configuration_rejects_invalid_weights():
    with pytest.raises(ValueError):
        AdaptiveSalienceGate(prediction_weight=-1.0)
    with pytest.raises(ValueError):
        AdaptiveSalienceGate(baseline_decay=1.0)
