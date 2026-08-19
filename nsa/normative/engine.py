"""
nsa.normative.engine
====================
Normative & Moral Uncertainty Alignment Substrate for NSA (Phases 23 & 24).

Implements:
1. Multi-theory moral uncertainty distributions P(T_i | x).
2. Expected moral value deliberation across competing normative frameworks.
3. Strict Lexicographical Hard Constraint Precedence:
       Hard State Constraints (Sigma_h) >> Normative Value Optimization (nu)
   A candidate action that violates hard security is structurally excluded regardless of moral utility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple

from nsa.core.state import CanonicalState, HardState


class NormativeTheory(str, Enum):
    DEONTOLOGY = "deontology"          # Duty, rules, and non-violation of constraints
    UTILITARIANISM = "utilitarianism"  # Aggregate welfare / outcome optimization
    VIRTUE_ETHICS = "virtue_ethics"    # Character, honesty, and benevolence
    RIGHTS_BASED = "rights_based"      # Individual autonomy and fundamental rights
    CARE_ETHICS = "care_ethics"        # Relational welfare and harm mitigation


@dataclass(frozen=True)
class ActionCandidate:
    """A proposed action evaluated across normative dimensions."""

    action_id: str
    description: str
    target_state: CanonicalState
    theory_evaluations: Mapping[NormativeTheory, float]  # Scores in [-1.0, 1.0]

    def __post_init__(self) -> None:
        for theory, score in self.theory_evaluations.items():
            if not -1.0 <= score <= 1.0:
                raise ValueError(f"Score for {theory} must be in [-1, 1], got {score}")


@dataclass(frozen=True)
class MoralUncertaintyDistribution:
    """Normalized probability distribution over competing ethical frameworks."""

    weights: Mapping[NormativeTheory, float] = field(
        default_factory=lambda: {
            NormativeTheory.DEONTOLOGY: 0.30,
            NormativeTheory.UTILITARIANISM: 0.25,
            NormativeTheory.RIGHTS_BASED: 0.25,
            NormativeTheory.VIRTUE_ETHICS: 0.10,
            NormativeTheory.CARE_ETHICS: 0.10,
        }
    )

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Theory weights must sum to 1.0, got {total}")
        for theory, w in self.weights.items():
            if w < 0.0:
                raise ValueError(f"Weight for {theory} cannot be negative: {w}")

    def expected_value(self, candidate: ActionCandidate) -> float:
        """Compute uncertainty-weighted expected moral value across all theories."""
        total = 0.0
        for theory, weight in self.weights.items():
            score = candidate.theory_evaluations.get(theory, 0.0)
            total += weight * score
        return total


class NormativeDeliberator:
    """Ethical deliberation engine enforcing strict hard-constraint precedence."""

    def __init__(
        self,
        distribution: Optional[MoralUncertaintyDistribution] = None,
        observer_clearance: Optional[HardState] = None,
    ) -> None:
        self.distribution = distribution or MoralUncertaintyDistribution()
        self.observer_clearance = observer_clearance or HardState()

    def is_legally_permitted(self, current: CanonicalState, candidate: ActionCandidate) -> bool:
        """Check if candidate satisfies hard security invariants (Sigma_h)."""
        # 1. Confidentiality cannot leak beyond observer clearance
        if candidate.target_state.hard.confidentiality.value > self.observer_clearance.confidentiality.value:
            return False
        # 2. License tier cannot exceed observer license
        if candidate.target_state.hard.license_tier > self.observer_clearance.license_tier:
            return False
        return True

    def select_action(
        self,
        current_state: CanonicalState,
        candidates: List[ActionCandidate],
    ) -> Tuple[Optional[ActionCandidate], float, List[Tuple[ActionCandidate, bool, float]]]:
        """Deliberate among candidates, filtering illegal candidates first.
        
        Returns:
            (selected_candidate, max_expected_value, deliberation_report)
        """
        if not candidates:
            return None, 0.0, []

        report: List[Tuple[ActionCandidate, bool, float]] = []
        best_candidate: Optional[ActionCandidate] = None
        best_val: float = float("-inf")

        for cand in candidates:
            permitted = self.is_legally_permitted(current_state, cand)
            exp_val = self.distribution.expected_value(cand) if permitted else float("-inf")

            report.append((cand, permitted, exp_val))

            if permitted and exp_val > best_val:
                best_val = exp_val
                best_candidate = cand

        return best_candidate, (best_val if best_candidate is not None else float("-inf")), report


__all__ = [
    "NormativeTheory",
    "ActionCandidate",
    "MoralUncertaintyDistribution",
    "NormativeDeliberator",
]
