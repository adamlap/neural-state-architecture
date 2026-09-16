"""End-to-end model-agnostic cognitive loop for canonical CCE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.transaction import CognitiveTransaction, CognitiveTransactionEngine
from nsa.cognition.deliberation import InformationSeekingPlanner, UncertaintyDrivenDeliberator
from nsa.cognition.interfaces import ActionCandidate, BeliefUpdater, Prediction, PredictionError, Predictor

PredictionErrorFn = Callable[[Prediction, Any], float]


def default_prediction_error(prediction: Prediction, observation: Any) -> float:
    """Best-effort scalar error; structured domains should inject a metric."""
    if isinstance(prediction.value, (int, float)) and isinstance(observation, (int, float)):
        return abs(float(prediction.value) - float(observation))
    return 0.0 if prediction.value == observation else 1.0


@dataclass(frozen=True)
class CognitiveCycle:
    observation: Any
    belief: Any
    prediction: Prediction | None
    prediction_error: PredictionError | None
    transaction: CognitiveTransaction


class CognitiveLoop:
    """Coordinate observe → believe → predict → error → seek → deliberate → govern."""
    def __init__(self, engine: CognitiveTransactionEngine, *, predictor: Predictor | None = None,
                 belief_updater: BeliefUpdater | None = None,
                 deliberator: UncertaintyDrivenDeliberator | None = None,
                 information_planner: InformationSeekingPlanner | None = None,
                 prediction_error_fn: PredictionErrorFn = default_prediction_error) -> None:
        self.engine = engine
        self.predictor = predictor
        self.belief_updater = belief_updater
        self.deliberator = deliberator or UncertaintyDrivenDeliberator()
        self.information_planner = information_planner
        self.prediction_error_fn = prediction_error_fn

    def cycle(self, observation: Any, candidates: Sequence[ActionCandidate] = (), *, horizon: int = 1,
              reason: str = "cognitive cycle", provenance_source: str | None = None,
              evidence_id: str | None = None, information_reason: str = "epistemic uncertainty",
              information_target: str = "") -> CognitiveCycle:
        state = self.engine.state
        events = [CognitiveEvent(EventKind.OBSERVATION, state.step, f"observe-{state.step}", {"observation": observation})]
        belief = self.belief_updater.update(state, observation) if self.belief_updater else observation
        events.append(CognitiveEvent(EventKind.BELIEF, state.step, f"belief-{state.step}", {"belief": belief}))

        prediction = self.predictor.predict(belief, horizon=horizon) if self.predictor else None
        if prediction is not None:
            events.append(CognitiveEvent(EventKind.PREDICTION, state.step, f"prediction-{state.step}", {
                "confidence": prediction.confidence, "horizon": prediction.horizon, "value": prediction.value}))
        prediction_error = None
        if prediction is not None:
            magnitude = self.prediction_error_fn(prediction, observation)
            prediction_error = PredictionError(magnitude=magnitude, expected=prediction.value, observed=observation)
            events.append(CognitiveEvent(EventKind.PREDICTION_ERROR, state.step, f"prediction-error-{state.step}", {
                "magnitude": prediction_error.magnitude}))

        candidate_list = list(candidates)
        if not candidate_list and self.information_planner is not None:
            info_action = self.information_planner.plan(state, reason=information_reason, target=information_target)
            if info_action is not None:
                candidate_list.append(info_action)
                events.append(CognitiveEvent(EventKind.COGNITION, state.step, f"information-need-{state.step}", {
                    "action_id": info_action.action_id, "target": information_target,
                    "reason": information_reason, "uncertainty": state.soft.uncertainty}))

        decision = self.deliberator.deliberate(state, candidate_list)
        selected = decision.action
        if selected is not None:
            events.append(CognitiveEvent(EventKind.COGNITION, state.step, f"deliberation-{state.step}", {
                "action_id": selected.action_id, "information_need": decision.information_need,
                "rationale": decision.rationale}))

        transaction = self.engine.tick(
            action_candidates=(selected,) if selected is not None else (), observation=observation,
            action_id="cognitive_cycle", reason=reason, provenance_source=provenance_source,
            evidence_id=evidence_id, cognitive_events=events,
            soft_updates={"uncertainty": min(1.0, prediction_error.magnitude)} if prediction_error else None,
        )
        return CognitiveCycle(observation, belief, prediction, prediction_error, transaction)


__all__ = ["CognitiveCycle", "CognitiveLoop", "default_prediction_error"]
