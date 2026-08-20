"""Live runtime trajectory records for Phase 19 self-model experiments.

The trajectory layer deliberately records only observations available at the
trusted runtime boundary. It never treats Ollama text as hidden-state access
and never gives the predictor authority over canonical hard state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nsa.runtime.typed_runtime import RuntimeGeneration
from nsa.self_state.model import SelfState


@dataclass(frozen=True)
class ActionFeatures:
    """Normalized features describing a generation request/result."""

    prompt_load: float
    max_tokens: float
    temperature: float
    output_load: float

    def as_list(self) -> list[float]:
        return [self.prompt_load, self.max_tokens, self.temperature, self.output_load]


@dataclass(frozen=True)
class TrajectoryStep:
    """One trusted runtime transition suitable for predictor training."""

    state_before: dict[str, float | int]
    action: ActionFeatures
    state_after: dict[str, float | int]
    observation_sources: dict[str, str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def build_action_features(
    *, prompt: str, max_tokens: int, temperature: float, output_text: str
) -> ActionFeatures:
    """Convert a live generation event to bounded predictor action features."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    return ActionFeatures(
        prompt_load=_clip01(len(prompt) / 4096.0),
        max_tokens=_clip01(max_tokens / 4096.0),
        temperature=_clip01(temperature / 2.0),
        output_load=_clip01(len(output_text) / float(max_tokens * 4)),
    )


def observe_runtime_self_state(
    *, before: SelfState, generation: RuntimeGeneration, max_tokens: int
) -> tuple[SelfState, dict[str, str]]:
    """Update only fields supported by live backend/runtime evidence.

    Ollama's HTTP API does not expose transformer confidence or hidden states.
    ``confidence`` therefore remains the backend's explicit estimate, while
    resource pressure is derived from Ollama's token counters when available.
    Unobservable fields are preserved rather than fabricated.
    """
    raw = generation.output.raw_response or {}
    eval_count = float(raw.get("eval_count", 0.0))
    prompt_eval_count = float(raw.get("prompt_eval_count", 0.0))
    denom = max(1.0, prompt_eval_count + float(max_tokens))
    resource_pressure = _clip01((prompt_eval_count + eval_count) / denom)
    confidence = _clip01(generation.output.confidence_estimate)

    after = before.observe(
        confidence=confidence,
        uncertainty=1.0 - confidence,
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
    generation: RuntimeGeneration,
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
    return TrajectoryStep(
        state_before=before.summary(),
        action=action,
        state_after=after.summary(),
        observation_sources=sources,
        model=model,
    )
