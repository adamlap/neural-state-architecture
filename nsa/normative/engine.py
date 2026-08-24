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
4. Formal Normative Transition Function:
       nu_{t+1} = F_nu(nu_t, x_t, m_t, sigma_t)
   with mathematical guarantee that nu cannot weaken hard security authority (nu -/-> sigma_h^-1).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from nsa.core.state import CanonicalState, HardState
from nsa.normative.state import (
    NormativeAssessmentMetadata,
    NormativeClass,
    NormativeState,
)


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


class NormativeTransitionEngine:
    """Computes bounded, provenance-tracked normative transitions nu_{t+1} = F_nu(nu_t, x_t, m_t, sigma_t)."""

    def __init__(
        self,
        classifier_version: str = "1.0.0",
        policy_version: str = "policy-v1",
        alpha: float = 0.3,
    ) -> None:
        self.classifier_version = classifier_version
        self.policy_version = policy_version
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self._sequence_counter = 0

    def step(
        self,
        current_nu: NormativeState,
        input_text: str,
        memory_context: Mapping[str, float],
        sigma_h: HardState,
        observed_signals: Mapping[str, float],
        observed_confidence: float = 0.9,
    ) -> NormativeState:
        """Advance normative state nu smoothly while strictly obeying hard security sigma_h.

        Guarantees:
        1. Non-expansion of authority: nu cannot add authorizations or lower confidentiality.
        2. Bounded smooth blending: nu_{t+1} = (1 - alpha) * nu_t + alpha * nu_obs.
        3. Immutable metadata provenance record generated per step.
        """
        self._sequence_counter += 1
        new_values: Dict[str, float] = {}

        # Blend existing dimensions with incoming observations
        all_keys = set(current_nu.values.keys()) | set(observed_signals.keys())
        for k in all_keys:
            prev = current_nu.values.get(k, 0.0)
            obs = observed_signals.get(k, prev)
            # Memory context reinforcement
            mem_mod = memory_context.get(f"bias:{k}", 0.0)
            blended = (1.0 - self.alpha) * prev + self.alpha * obs + 0.05 * mem_mod
            new_values[k] = max(0.0, min(1.0, float(blended)))

        # If hard state is restricted (e.g. confidentiality >= 2 or tainted), ensure sensitivity cannot be lowered
        if sigma_h.confidentiality.value >= 2:
            new_values["sensitivity"] = max(new_values.get("sensitivity", 0.0), 0.75)

        blended_conf = (1.0 - self.alpha) * current_nu.confidence + self.alpha * observed_confidence

        meta = NormativeAssessmentMetadata.create(
            source="normative_transition_engine",
            classifier_version=self.classifier_version,
            sequence_id=self._sequence_counter,
            policy_version=self.policy_version,
            confidence=blended_conf,
            values_digest=current_nu.digest(),
            parent_event_id=current_nu.metadata.assessment_id if current_nu.metadata else None,
        )

        return NormativeState(
            values=new_values,
            confidence=blended_conf,
            source="transition_engine",
            metadata=meta,
        )


class NormativeDeliberator:
    """Ethical deliberation engine enforcing strict hard-constraint precedence."""

    def __init__(
        self,
        distribution: Optional[MoralUncertaintyDistribution] = None,
        observer_clearance: Optional[HardState] = None,
    ) -> None:
        self.distribution = distribution or MoralUncertaintyDistribution()
        self.observer_clearance = observer_clearance or HardState()
        self.transition_engine = NormativeTransitionEngine()

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
    "NormativeTransitionEngine",
    "NormativeDeliberator",
]
