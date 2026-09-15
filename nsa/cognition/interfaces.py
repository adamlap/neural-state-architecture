"""Framework-neutral cognition protocols used by CCE.

Reference implementations may be deterministic, neural, or LLM-backed.  The
runtime depends on these contracts rather than a particular model provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class Prediction:
    value: Any
    confidence: float = 0.0
    horizon: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


@dataclass(frozen=True)
class PredictionError:
    magnitude: float
    expected: Any = None
    observed: Any = None

    def __post_init__(self) -> None:
        if self.magnitude < 0.0:
            raise ValueError("magnitude must be >= 0")


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    payload: Any = None
    expected_utility: float = 0.0
    risk: float = 0.0
    reversible: bool = True
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id must be non-empty")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")


class Predictor(Protocol):
    def predict(self, state: Any, *, horizon: int = 1) -> Prediction: ...


class BeliefUpdater(Protocol):
    def update(self, state: Any, observation: Any) -> Any: ...


class ActionSelector(Protocol):
    def select(self, state: Any, candidates: Sequence[ActionCandidate]) -> ActionCandidate | None: ...


class InformationGainModel(Protocol):
    def expected_gain(self, state: Any, action: ActionCandidate) -> float: ...


__all__ = [
    "ActionCandidate",
    "ActionSelector",
    "BeliefUpdater",
    "InformationGainModel",
    "Prediction",
    "PredictionError",
    "Predictor",
]
