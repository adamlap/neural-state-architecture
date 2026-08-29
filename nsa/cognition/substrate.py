"""Deterministic cognitive substrate for the public NSA/CCE runtime.

This module implements consciousness-inspired computational mechanisms without
making a claim that any resulting system is conscious.  It is deliberately
model-agnostic and dependency-free: the model is a replaceable source of
observations/actions, while this substrate owns explicit cognitive state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import log2
from typing import Any, Mapping, Sequence


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _entropy(values: Sequence[float]) -> float:
    total = sum(max(0.0, x) for x in values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    return sum(-(x / total) * log2(x / total) for x in values if x > 0) / log2(len(values))


@dataclass(frozen=True)
class CognitiveSwitches:
    workspace: bool = True
    recurrence: bool = True
    predictive_processing: bool = True
    self_model: bool = True
    integration: bool = True


@dataclass(frozen=True)
class WorkspaceCandidate:
    key: str
    content: Any
    salience: float
    confidence: float
    novelty: float = 0.0
    source: str = "unknown"

    @property
    def competition_score(self) -> float:
        return _clip(0.45 * self.salience + 0.30 * self.confidence + 0.25 * self.novelty)


@dataclass(frozen=True)
class WorkspaceState:
    capacity: int = 4
    active: tuple[str, ...] = ()
    ignition_count: int = 0
    broadcast_count: int = 0
    broadcast_history: tuple[str, ...] = ()
    last_ignition: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    key: str
    value: float
    precision: float = 1.0
    age: int = 0


@dataclass(frozen=True)
class PredictionState:
    predictions: tuple[Prediction, ...] = ()
    total_error: float = 0.0
    mean_error: float = 0.0
    uncertainty: float = 0.0
    update_count: int = 0


@dataclass(frozen=True)
class IntegrationGraph:
    nodes: tuple[str, ...] = ()
    edges: tuple[tuple[str, str, float], ...] = ()
    integration: float = 0.0
    causal_influence: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class SelfModelState:
    internal_state: Mapping[str, float] = field(default_factory=dict)
    predicted_self: Mapping[str, float] = field(default_factory=dict)
    prediction_error: float = 0.0
    confidence: float = 0.0
    metacognitive_signal: float = 0.0
    update_count: int = 0


@dataclass(frozen=True)
class CognitiveMetrics:
    workspace_ignitions: int = 0
    broadcast_coverage: float = 0.0
    prediction_error: float = 0.0
    integration: float = 0.0
    recurrence: float = 0.0
    self_model_accuracy: float = 0.0
    cognitive_continuity: float = 0.0
    information_gain: float = 0.0


@dataclass(frozen=True)
class CognitiveState:
    cycle: int = 0
    workspace: WorkspaceState = field(default_factory=WorkspaceState)
    prediction: PredictionState = field(default_factory=PredictionState)
    integration: IntegrationGraph = field(default_factory=IntegrationGraph)
    self_model: SelfModelState = field(default_factory=SelfModelState)
    metrics: CognitiveMetrics = field(default_factory=CognitiveMetrics)
    last_observation: Any = None
    last_action: Any = None
    recurrence_trace: tuple[str, ...] = ()
    switches: CognitiveSwitches = field(default_factory=CognitiveSwitches)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CognitiveSubstrate:
    """Pure state transition for the Go 1 cognitive substrate."""

    def __init__(self, *, switches: CognitiveSwitches | None = None, workspace_capacity: int = 4) -> None:
        if workspace_capacity < 1:
            raise ValueError("workspace_capacity must be >= 1")
        self.switches = switches or CognitiveSwitches()
        self.workspace_capacity = workspace_capacity

    @staticmethod
    def information_gain(prior_uncertainty: float, observation_confidence: float) -> float:
        """Approximate normalized information gain from a new observation."""
        return _clip(_clip(prior_uncertainty) * _clip(observation_confidence))

    def _candidates(self, observation: Any, confidence: float) -> list[WorkspaceCandidate]:
        text = str(observation)
        words = [w for w in text.split() if w]
        novelty = 1.0 if not words else min(1.0, len(set(words)) / max(1, len(words)))
        return [
            WorkspaceCandidate("observation", observation, salience=_clip(confidence), confidence=_clip(confidence), novelty=novelty, source="perception"),
            WorkspaceCandidate("prediction_error", self._prediction_error(self._state_prediction, observation), salience=0.8, confidence=0.8, novelty=novelty, source="prediction"),
        ]

    @staticmethod
    def _prediction_error(prediction: PredictionState, observation: Any) -> float:
        if not prediction.predictions:
            return 1.0
        try:
            observed = float(observation)
        except (TypeError, ValueError):
            return prediction.mean_error
        return min(1.0, min(abs(observed - p.value) * max(0.0, p.precision) for p in prediction.predictions))

    def _select_workspace(self, candidates: Sequence[WorkspaceCandidate], previous: WorkspaceState) -> WorkspaceState:
        if not self.switches.workspace:
            return replace(previous, active=(), last_ignition=())
        selected = sorted(candidates, key=lambda c: (-c.competition_score, c.key))[:previous.capacity]
        keys = tuple(c.key for c in selected)
        ignited = keys != previous.active
        history = (previous.broadcast_history + keys)[-64:]
        return replace(
            previous,
            active=keys,
            ignition_count=previous.ignition_count + int(ignited),
            broadcast_count=previous.broadcast_count + int(bool(keys)),
            broadcast_history=history,
            last_ignition=keys if ignited else (),
        )

    def _update_prediction(self, previous: PredictionState, observation: Any) -> PredictionState:
        if not self.switches.predictive_processing:
            return previous
        try:
            observed = float(observation)
        except (TypeError, ValueError):
            error = previous.mean_error
            value = None
        else:
            value = observed
            error = 1.0 if not previous.predictions else min(1.0, min(abs(observed - p.value) for p in previous.predictions))
        updated = []
        if value is not None:
            if previous.predictions:
                for p in previous.predictions:
                    updated.append(replace(p, value=p.value + 0.25 * (value - p.value), age=0))
            else:
                updated.append(Prediction("observation", value, precision=1.0))
        else:
            updated = [replace(p, age=p.age + 1) for p in previous.predictions]
        mean = _clip(0.75 * previous.mean_error + 0.25 * error)
        return replace(previous, predictions=tuple(updated), total_error=previous.total_error + error, mean_error=mean, uncertainty=mean, update_count=previous.update_count + 1)

    def _integrate(self, previous: IntegrationGraph, workspace: WorkspaceState, prediction: PredictionState, self_model: SelfModelState) -> IntegrationGraph:
        if not self.switches.integration:
            return previous
        nodes = ("perception", "prediction", "attention", "workspace", "self")
        edges = tuple((a, b, 1.0) for a, b in (("perception", "prediction"), ("prediction", "attention"), ("attention", "workspace"), ("workspace", "self"), ("self", "prediction")))
        active_factor = len(workspace.active) / max(1, workspace.capacity)
        integration = _clip(0.30 * active_factor + 0.30 * (1.0 - prediction.mean_error) + 0.40 * self_model.confidence)
        return replace(previous, nodes=nodes, edges=edges, integration=integration, causal_influence=edges)

    def _update_self_model(self, previous: SelfModelState, state: CognitiveState, observation: Any) -> SelfModelState:
        if not self.switches.self_model:
            return previous
        internal = {
            "prediction_error": state.prediction.mean_error,
            "workspace_load": len(state.workspace.active) / max(1, state.workspace.capacity),
            "integration": state.integration.integration,
        }
        predicted = previous.predicted_self or internal
        error = sum(abs(internal[k] - predicted.get(k, 0.0)) for k in internal) / len(internal)
        accuracy = _clip(1.0 - error)
        signal = _clip(0.5 * error + 0.5 * (1.0 - accuracy))
        return SelfModelState(internal_state=internal, predicted_self=internal, prediction_error=error, confidence=accuracy, metacognitive_signal=signal, update_count=previous.update_count + 1)

    def transition(self, state: CognitiveState, observation: Any, *, confidence: float = 1.0, action: Any = None) -> CognitiveState:
        """Advance one cognitive cycle from an observation; no wall-clock polling."""
        self._state_prediction = state.prediction
        prediction = self._update_prediction(state.prediction, observation)
        candidates = self._candidates(observation, confidence)
        workspace = self._select_workspace(candidates, state.workspace)
        provisional = replace(state, prediction=prediction, workspace=workspace, last_observation=observation, last_action=action)
        self_model = self._update_self_model(state.self_model, provisional, observation)
        integration = self._integrate(state.integration, workspace, prediction, self_model)
        recurrence = _clip(1.0 - abs(integration.integration - state.integration.integration)) if self.switches.recurrence else 0.0
        gain = self.information_gain(prediction.uncertainty, confidence)
        metrics = CognitiveMetrics(
            workspace_ignitions=workspace.ignition_count,
            broadcast_coverage=_clip(len(workspace.active) / max(1, workspace.capacity)),
            prediction_error=prediction.mean_error,
            integration=integration.integration,
            recurrence=recurrence,
            self_model_accuracy=self_model.confidence,
            cognitive_continuity=_clip(0.5 * recurrence + 0.5 * (1.0 - prediction.mean_error)),
            information_gain=gain,
        )
        trace = (state.recurrence_trace + ("prediction", "attention", "workspace", "integration", "self_model"))[-64:]
        return CognitiveState(cycle=state.cycle + 1, workspace=workspace, prediction=prediction, integration=integration, self_model=self_model, metrics=metrics, last_observation=observation, last_action=action, recurrence_trace=trace, switches=self.switches)


__all__ = ["CognitiveMetrics", "CognitiveState", "CognitiveSubstrate", "CognitiveSwitches", "IntegrationGraph", "Prediction", "PredictionState", "SelfModelState", "WorkspaceCandidate", "WorkspaceState"]
