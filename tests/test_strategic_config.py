import pytest

from nsa.core.state import CanonicalState
from nsa.cognition.interfaces import PredictionError
from nsa.strategy import (
    ScenarioOutcome,
    StrategicAttentionBias,
    StrategicConfig,
    StrategicController,
    StrategicFeedback,
    StrategicField,
    StrategyCandidate,
)


def _candidate():
    return (
        StrategyCandidate(
            "safe",
            (ScenarioOutcome("ok", 1.0, goal_progress=1.0),),
            prior=0.5,
        ),
    )


def test_snm_can_be_disabled_without_evaluating_candidates():
    controller = StrategicController(config=StrategicConfig(enabled=False))
    evaluation = controller.evaluate(CanonicalState(), ())
    assert evaluation.field.active_strategy is None
    assert evaluation.distribution.probabilities == {}
    assert evaluation.field.strength == 0.0


def test_disabled_attention_is_an_exact_zero_bias():
    controller = StrategicController(
        config=StrategicConfig(enabled=True, attention_enabled=False)
    )
    evaluation = controller.evaluate(CanonicalState(), _candidate())
    assert controller.attention_bias(evaluation, [{"safe": 1.0}]) == (0.0,)


def test_feedback_can_be_disabled_independently():
    candidates = _candidate()
    feedback = StrategicFeedback(
        "safe",
        PredictionError(magnitude=1.0),
        observed_goal_progress=1.0,
        observed_risk=0.0,
    )
    controller = StrategicController(
        config=StrategicConfig(enabled=True, feedback_enabled=False)
    )
    assert controller.update_candidates(candidates, feedback) == candidates


def test_environment_configuration_can_disable_snm(monkeypatch):
    monkeypatch.setenv("TEST_SNM_ENABLED", "false")
    monkeypatch.setenv("TEST_SNM_FEEDBACK_ENABLED", "0")
    monkeypatch.setenv("TEST_SNM_ATTENTION_ENABLED", "off")
    monkeypatch.setenv("TEST_SNM_MAX_BIAS", "0.1")
    config = StrategicConfig.from_env("TEST_SNM_")
    assert config == StrategicConfig(
        enabled=False,
        feedback_enabled=False,
        attention_enabled=False,
        max_bias=0.1,
    )


def test_environment_configuration_defaults_are_enabled(monkeypatch):
    for name in (
        "TEST_SNM_ENABLED",
        "TEST_SNM_FEEDBACK_ENABLED",
        "TEST_SNM_ATTENTION_ENABLED",
        "TEST_SNM_MAX_BIAS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert StrategicConfig.from_env("TEST_SNM_") == StrategicConfig()


def test_invalid_environment_boolean_is_rejected(monkeypatch):
    monkeypatch.setenv("TEST_SNM_ENABLED", "sometimes")
    with pytest.raises(ValueError):
        StrategicConfig.from_env("TEST_SNM_")


def test_torch_adapter_can_be_disabled():
    torch = pytest.importorskip("torch")
    field = StrategicField({"safe": 1.0}, "safe", confidence=1.0)
    adapter = StrategicAttentionBias.from_config(
        StrategicConfig(attention_enabled=False)
    )
    logits = torch.zeros(1, 1, 1, 1)
    bias = adapter.build(logits, field, [{"safe": 1.0}])
    assert bias.shape == (1, 1, 1, 1)
    assert float(bias.sum()) == 0.0
