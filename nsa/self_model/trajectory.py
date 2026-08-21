"""Live runtime trajectory records for Phase 19 self-model experiments.

Only observations available at the trusted runtime boundary are recorded.
Ollama text is never treated as hidden-state access and prediction has no
authority over canonical hard state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from nsa.self_state.model import SelfState

if TYPE_CHECKING:
    from nsa.runtime.typed_runtime import RuntimeGeneration


@dataclass(frozen=True)
class ActionFeatures:
    prompt_load: float
    max_tokens: float
    temperature: float
    output_load: float

    def as_list(self) -> list[float]:
        return [self.prompt_load, self.max_tokens, self.temperature, self.output_load]


@dataclass(frozen=True)
class TrajectoryStep:
    state_before: dict[str, float | int]
    action: ActionFeatures
    state_after: dict[str, float | int]
    observation_sources: dict[str, str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _bounded_complement(value: float) -> float:
    """Return a deterministic decimal representation of ``1 - value``.

    Self-state values are a bounded public schema, so normalization here avoids
    binary floating-point artifacts such as ``0.19999999999999996`` while
    retaining the underlying observation rather than changing its semantics.
    """
    return _clip01(round(1.0 - float(value), 12))


def build_action_features(
    *, prompt: str, max_tokens: int, temperature: float, output_text: str
) -> ActionFeatures:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return ActionFeatures(
        _clip01(len(prompt) / 4096.0),
        _clip01(max_tokens / 4096.0),
        _clip01(temperature / 2.0),
        _clip01(len(output_text) / float(max_tokens * 4)),
    )


def observe_runtime_self_state(
    *, before: SelfState, generation: "RuntimeGeneration", max_tokens: int
) -> tuple[SelfState, dict[str, str]]:
    raw = generation.output.raw_response or {}
    eval_count = float(raw.get("eval_count", 0.0))
    prompt_eval_count = float(raw.get("prompt_eval_count", 0.0))
    denom = max(1.0, prompt_eval_count + float(max_tokens))
    resource_pressure = _clip01((prompt_eval_count + eval_count) / denom)
    confidence = _clip01(generation.output.confidence_estimate)
    after = before.observe(
        confidence=confidence,
        uncertainty=_bounded_complement(confidence),
        resource_pressure=resource_pressure,
    )
    sources = {
        "confidence": "ollama-backend-explicit-estimate",
        "uncertainty": "derived-as-one-minus-confidence",
        "resource_pressure": "ollama-eval-token-counters",
        "perceived_risk": "preserved-unobservable",
        "capability_awareness": "preserved-unobservable",
        "goal_progress": "preserved-unobservable",
        "state_prediction_error": "preserved-until-predictor-comparison",
    }
    return after, sources


def trajectory_step(
    *,
    before: SelfState,
    generation: "RuntimeGeneration",
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str,
) -> TrajectoryStep:
    action = build_action_features(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        output_text=generation.output.text,
    )
    after, sources = observe_runtime_self_state(
        before=before, generation=generation, max_tokens=max_tokens
    )
    return TrajectoryStep(before.summary(), action, after.summary(), sources, model)
