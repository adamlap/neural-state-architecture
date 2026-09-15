"""Model-agnostic deliberation primitives for autonomous information seeking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nsa.cognition.interfaces import ActionCandidate, InformationGainModel
from nsa.core.state import CanonicalState


@dataclass(frozen=True)
class DeliberationDecision:
    action: ActionCandidate | None
    information_need: float
    rationale: str


class UncertaintyDrivenDeliberator:
    """Prefer evidence acquisition when epistemic uncertainty dominates utility."""

    def __init__(self, *, information_threshold: float = 0.65, gain_weight: float = 0.5) -> None:
        if not 0.0 <= information_threshold <= 1.0:
            raise ValueError("information_threshold must be in [0, 1]")
        if gain_weight < 0.0:
            raise ValueError("gain_weight must be >= 0")
        self.information_threshold = information_threshold
        self.gain_weight = gain_weight

    def deliberate(
        self,
        state: CanonicalState,
        candidates: Sequence[ActionCandidate],
        *,
        information_model: InformationGainModel | None = None,
    ) -> DeliberationDecision:
        if not candidates:
            return DeliberationDecision(None, state.soft.uncertainty, "no candidate actions")

        best = None
        best_score = float("-inf")
        for candidate in candidates:
            gain = information_model.expected_gain(state, candidate) if information_model else 0.0
            score = candidate.expected_utility - candidate.risk + self.gain_weight * gain
            if state.soft.uncertainty >= self.information_threshold:
                score += self.gain_weight * gain
            if score > best_score:
                best_score, best = score, candidate

        if state.soft.uncertainty >= self.information_threshold and best is not None:
            return DeliberationDecision(best, state.soft.uncertainty, "high uncertainty: information value included in action selection")
        return DeliberationDecision(best, state.soft.uncertainty, "ordinary utility/risk deliberation")


__all__ = ["DeliberationDecision", "UncertaintyDrivenDeliberator"]
