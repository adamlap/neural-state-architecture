"""Typed state for the NSA strategic control layer."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Tuple

def _bounded(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return float(value)

@dataclass(frozen=True)
class ScenarioOutcome:
    outcome_id: str
    probability: float
    goal_progress: float
    cost: float = 0.0
    risk: float = 0.0
    time: float = 0.0
    def __post_init__(self) -> None:
        for name in ("probability","goal_progress","cost","risk","time"):
            _bounded(name, getattr(self, name))

@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    outcomes: Tuple[ScenarioOutcome, ...]
    prior: float = 0.0
    information_gain: float = 0.0
    reversibility: float = 1.0
    def __post_init__(self) -> None:
        if not self.strategy_id or not self.outcomes:
            raise ValueError("strategy_id and at least one outcome are required")
        for name in ("prior","information_gain","reversibility"):
            _bounded(name, getattr(self, name))
        total = sum(o.probability for o in self.outcomes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"outcome probabilities must sum to 1.0, got {total:.6f}")

@dataclass(frozen=True)
class StrategyDistribution:
    probabilities: Mapping[str, float]
    selected: str
    confidence: float
    expected_value: float
    def __post_init__(self) -> None:
        if not self.probabilities or self.selected not in self.probabilities:
            raise ValueError("selected strategy must be present in probabilities")
        _bounded("confidence", self.confidence)

@dataclass(frozen=True)
class StrategicField:
    """Compact, bounded soft control signal; never an authority state."""
    strategy_weights: Mapping[str, float] = field(default_factory=dict)
    active_strategy: str | None = None
    confidence: float = 0.0
    urgency: float = 0.0
    risk_sensitivity: float = 0.5
    horizon: float = 0.5
    generation: int = 0
    def __post_init__(self) -> None:
        for name in ("confidence","urgency","risk_sensitivity","horizon"):
            _bounded(name, getattr(self, name))
        for key, value in self.strategy_weights.items():
            if not key:
                raise ValueError("strategy weight keys must be non-empty")
            _bounded(f"strategy_weights[{key}]", value)
    @property
    def strength(self) -> float:
        return min(1.0, max(0.0, self.confidence * (0.5 + 0.5 * self.urgency)))
    def as_mapping(self) -> Mapping[str, object]:
        return {"strategy_weights": dict(self.strategy_weights),
                "active_strategy": self.active_strategy,
                "confidence": self.confidence, "urgency": self.urgency,
                "risk_sensitivity": self.risk_sensitivity,
                "horizon": self.horizon, "generation": self.generation}
