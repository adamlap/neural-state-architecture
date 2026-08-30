"""Embodied and active-cognition extensions for the NSA cognitive substrate."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

def _clip(value: float) -> float: return max(0.0, min(1.0, float(value)))

@dataclass(frozen=True)
class HomeostaticState:
    energy: float = 1.0
    uncertainty: float = 0.0
    arousal: float = 0.0
    stability: float = 1.0

@dataclass(frozen=True)
class ValuationState:
    preferred: Mapping[str, float] | None = None
    last_action: str | None = None
    last_value: float = 0.0
    def __post_init__(self) -> None:
        if self.preferred is None: object.__setattr__(self, "preferred", {})

@dataclass(frozen=True)
class IdentityState:
    identity_id: str = "nsa-agent"
    autobiographical: tuple[str, ...] = ()
    commitments: tuple[str, ...] = ()
    continuity: float = 1.0
    age: int = 0

@dataclass(frozen=True)
class ActiveCognitionState:
    cycle: int = 0
    homeostasis: HomeostaticState = HomeostaticState()
    valuation: ValuationState = ValuationState()
    identity: IdentityState = IdentityState()
    selected_action: str | None = None
    information_gain: float = 0.0
    expected_free_energy: float = 0.0
    action_scores: tuple[tuple[str, float], ...] = ()
    def to_dict(self) -> dict[str, Any]:
        return {"cycle": self.cycle, "homeostasis": {"energy": self.homeostasis.energy, "uncertainty": self.homeostasis.uncertainty, "arousal": self.homeostasis.arousal, "stability": self.homeostasis.stability}, "valuation": {"preferred": dict(self.valuation.preferred or {}), "last_action": self.valuation.last_action, "last_value": self.valuation.last_value}, "identity": {"identity_id": self.identity.identity_id, "autobiographical": self.identity.autobiographical, "commitments": self.identity.commitments, "continuity": self.identity.continuity, "age": self.identity.age}, "selected_action": self.selected_action, "information_gain": self.information_gain, "expected_free_energy": self.expected_free_energy, "action_scores": self.action_scores}

class ActiveCognition:
    """Deterministic action/observation selection; never grants authority."""
    def score_actions(self, actions: Sequence[str], *, uncertainty: float, information_gain: Mapping[str, float] | None = None, expected_utility: Mapping[str, float] | None = None, risk: Mapping[str, float] | None = None, epistemic_weight: float = 1.0, utility_weight: float = 1.0, risk_weight: float = 1.0) -> tuple[tuple[str, float], ...]:
        ig, utility, risks = information_gain or {}, expected_utility or {}, risk or {}
        scored = []
        for action in actions:
            gain = _clip(ig.get(action, 0.0)); r = _clip(risks.get(action, 0.0)); u = utility.get(action, 0.0)
            score = utility_weight * u + epistemic_weight * gain - risk_weight * r - _clip(uncertainty - gain)
            scored.append((action, score))
        return tuple(sorted(scored, key=lambda x: (-x[1], x[0])))

    def transition(self, state: ActiveCognitionState, *, uncertainty: float, candidate_actions: Sequence[str] = (), information_gain: Mapping[str, float] | None = None, expected_utility: Mapping[str, float] | None = None, risk: Mapping[str, float] | None = None, observation: Any = None, chosen_action: str | None = None) -> ActiveCognitionState:
        scores = self.score_actions(candidate_actions, uncertainty=uncertainty, information_gain=information_gain, expected_utility=expected_utility, risk=risk)
        action = chosen_action or (scores[0][0] if scores else None); gains = information_gain or {}; gain = _clip(gains.get(action, 0.0)) if action else 0.0
        free_energy = _clip(uncertainty - gain); energy = _clip(state.homeostasis.energy - (0.01 if action else 0.0) + 0.005 * gain); stability = _clip(1.0 - 0.5 * free_energy); arousal = _clip(0.5 * uncertainty + 0.5 * (1.0 - stability))
        preferred = dict(state.valuation.preferred or {})
        if action: preferred[action] = 0.9 * preferred.get(action, 0.0) + 0.1 * (1.0 - free_energy)
        autobiography = state.identity.autobiographical
        if observation is not None: autobiography = (autobiography + (str(observation)[:160],))[-128:]
        continuity = _clip(0.8 * state.identity.continuity + 0.2 * stability)
        identity = replace(state.identity, autobiographical=autobiography, continuity=continuity, age=state.identity.age + 1)
        valuation = ValuationState(preferred=preferred, last_action=action, last_value=preferred.get(action, 0.0) if action else 0.0)
        return ActiveCognitionState(cycle=state.cycle + 1, homeostasis=HomeostaticState(energy=energy, uncertainty=_clip(uncertainty), arousal=arousal, stability=stability), valuation=valuation, identity=identity, selected_action=action, information_gain=gain, expected_free_energy=free_energy, action_scores=scores)

__all__ = ["ActiveCognition", "ActiveCognitionState", "HomeostaticState", "IdentityState", "ValuationState"]
