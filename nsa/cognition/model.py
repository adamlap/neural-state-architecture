"""Provider-neutral cognitive model contracts.

LLMs and other learned systems are proposal engines. They may suggest belief,
prediction, information-seeking, goal, and action changes, but they never own
canonical state or execute effects. The CCE remains the authority boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from nsa.cognition.interfaces import ActionCandidate, Prediction


@dataclass(frozen=True)
class InformationNeedProposal:
    """A model's request for evidence, expressed without granting authority."""

    question: str
    expected_information_gain: float = 0.0
    urgency: float = 0.0
    preferred_capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must be non-empty")
        if not 0.0 <= self.expected_information_gain <= 1.0:
            raise ValueError("expected_information_gain must be in [0, 1]")
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError("urgency must be in [0, 1]")


@dataclass(frozen=True)
class CognitiveProposal:
    """Complete model proposal consumed by the governed CCE layer."""

    belief_updates: tuple[Mapping[str, Any], ...] = ()
    predictions: tuple[Prediction, ...] = ()
    prediction_errors: tuple[Mapping[str, Any], ...] = ()
    information_needs: tuple[InformationNeedProposal, ...] = ()
    goal_updates: tuple[Mapping[str, Any], ...] = ()
    action_candidates: tuple[ActionCandidate, ...] = ()
    rationale: str = ""
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def has_action(self) -> bool:
        return bool(self.action_candidates)

    @property
    def has_information_need(self) -> bool:
        return bool(self.information_needs)


@dataclass(frozen=True)
class CognitiveContext:
    """Provider-neutral context presented to a cognitive model."""

    state: Any
    observations: Sequence[Any] = ()
    tools: Sequence[Any] = ()
    system: str | None = None
    task: str = "cognition"
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CognitiveModel(Protocol):
    """Interface implemented by OpenAI, Gemini, local and future providers."""

    async def propose(self, context: CognitiveContext) -> CognitiveProposal:
        """Return a proposal; never mutate canonical state or execute effects."""
        ...


__all__ = [
    "CognitiveContext",
    "CognitiveModel",
    "CognitiveProposal",
    "InformationNeedProposal",
]
