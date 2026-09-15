"""Model-agnostic deliberation and autonomous information seeking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from nsa.cognition.interfaces import ActionCandidate, InformationGainModel
from nsa.core.state import CanonicalState


@dataclass(frozen=True)
class DeliberationDecision:
    action: ActionCandidate | None
    information_need: float
    rationale: str


@dataclass(frozen=True)
class InformationNeed:
    """Explicit representation of missing evidence that cognition can pursue."""
    magnitude: float
    reason: str
    target: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError("magnitude must be in [0, 1]")
        if not self.reason:
            raise ValueError("reason must be non-empty")


class InformationSeekingPlanner:
    """Generate governed T1 evidence-gathering candidates from uncertainty.

    The generated action still enters the ordinary policy/capability/safety
    transaction path; information gathering is not an authority bypass.
    """
    def __init__(self, *, threshold: float = 0.65, action_factory: Callable[[InformationNeed], ActionCandidate] | None = None) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold
        self.action_factory = action_factory

    def identify_need(self, state: CanonicalState, *, reason: str = "epistemic uncertainty", target: str = "") -> InformationNeed | None:
        if state.soft.uncertainty < self.threshold:
            return None
        return InformationNeed(state.soft.uncertainty, reason, target)

    def plan(self, state: CanonicalState, *, reason: str = "epistemic uncertainty", target: str = "") -> ActionCandidate | None:
        need = self.identify_need(state, reason=reason, target=target)
        if need is None or self.action_factory is None:
            return None
        return self.action_factory(need)


class UncertaintyDrivenDeliberator:
    """Prefer evidence acquisition as uncertainty rises without double weighting."""
    def __init__(self, *, information_threshold: float = 0.65, gain_weight: float = 0.5) -> None:
        if not 0.0 <= information_threshold <= 1.0:
            raise ValueError("information_threshold must be in [0, 1]")
        if gain_weight < 0.0:
            raise ValueError("gain_weight must be >= 0")
        self.information_threshold = information_threshold
        self.gain_weight = gain_weight

    def deliberate(self, state: CanonicalState, candidates: Sequence[ActionCandidate], *, information_model: InformationGainModel | None = None) -> DeliberationDecision:
        if not candidates:
            return DeliberationDecision(None, state.soft.uncertainty, "no candidate actions")
        best = None
        best_score = float("-inf")
        # Information value receives a single uncertainty-scaled contribution.
        scale = max(0.0, (state.soft.uncertainty - self.information_threshold) / max(1e-9, 1.0 - self.information_threshold))
        for candidate in candidates:
            gain = information_model.expected_gain(state, candidate) if information_model else 0.0
            score = candidate.expected_utility - candidate.risk + self.gain_weight * gain * (1.0 + scale)
            if score > best_score:
                best_score, best = score, candidate
        rationale = "uncertainty-scaled information value included" if scale > 0 else "ordinary utility/risk deliberation"
        return DeliberationDecision(best, state.soft.uncertainty, rationale)


__all__ = ["DeliberationDecision", "InformationNeed", "InformationSeekingPlanner", "UncertaintyDrivenDeliberator"]
