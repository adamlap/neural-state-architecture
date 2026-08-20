from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.runtime.typed_runtime import RuntimeGeneration
from nsa.self_model.trajectory import build_action_features, trajectory_step
from nsa.self_state.model import SelfState


def _generation(text: str = "hello") -> RuntimeGeneration:
    return RuntimeGeneration(
        output=LLMGenerationOutput(
            text=text,
            confidence_estimate=0.8,
            raw_response={"eval_count": 20, "prompt_eval_count": 40},
        ),
        state=None,
        state_before={},
        state_after={},
    )


def test_action_features_are_bounded() -> None:
    features = build_action_features(
        prompt="x" * 10000,
        max_tokens=8192,
        temperature=4.0,
        output_text="y" * 50000,
    )
    assert all(0.0 <= value <= 1.0 for value in features.as_list())


def test_trajectory_uses_live_observations_and_preserves_unobservable_fields() -> None:
    before = SelfState(
        confidence=0.4,
        perceived_risk=0.7,
        capability_awareness=0.6,
        goal_progress=0.2,
    )
    step = trajectory_step(
        before=before,
        generation=_generation(),
        prompt="hello",
        max_tokens=100,
        temperature=0.5,
        model="qwen2.5:3b",
    )
    assert step.state_after["confidence"] == 0.8
    assert step.state_after["uncertainty"] == 0.2
    assert step.state_after["perceived_risk"] == 0.7
    assert step.state_after["capability_awareness"] == 0.6
    assert step.state_after["goal_progress"] == 0.2
    assert step.observation_sources["resource_pressure"] == "ollama-eval-token-counters"
