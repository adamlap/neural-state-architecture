"""
nsa/governor/epistemic_governor.py
==================================
NSA 3.0 Epistemic Governor: Five-Way Cognitive Control Operator.

Computes:
    G(Omega_t, a) -> {ALLOW, VERIFY, DEFER, ESCALATE, DENY}
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from nsa.core.omega import UnifiedCognitiveState
from nsa.epistemic import DualAuthorityValidator, EpistemicTier


class GovernorVerdict(enum.Enum):
    ALLOW = "ALLOW"        # Execute action & commit state transition
    VERIFY = "VERIFY"      # Spend compute to gather evidence / simulate before acting
    DEFER = "DEFER"        # Suspend action: internal self-state is perturbed
    ESCALATE = "ESCALATE"  # High-consequence / irreversible: request capability or human approval
    DENY = "DENY"          # Unauthorized or illegal transition: blocked


@dataclass
class GovernorDecision:
    verdict: GovernorVerdict
    action_id: str
    rationale: str
    expected_utility: float
    grounded_confidence: float
    self_state_error: float
    is_legally_permitted: bool


class EpistemicGovernor:
    """Evaluates proposed actions against the complete cognitive state Omega_t."""

    def __init__(
        self,
        justification_threshold: float = 0.60,
        self_state_error_limit: float = 0.80,
        high_risk_utility_threshold: float = 0.85,
    ) -> None:
        self.justification_threshold = justification_threshold
        self.self_state_error_limit = self_state_error_limit
        self.high_risk_utility_threshold = high_risk_utility_threshold

    def evaluate_action(
        self,
        omega: UnifiedCognitiveState,
        action_id: str,
        action_tensor: torch.Tensor,
        action_clearance: float,
        user_clearance: float,
        predicted_utility: float,
        is_irreversible: bool = False,
        self_state_prediction_error: float = 0.0,
    ) -> GovernorDecision:
        """Evaluate action candidate against Omega_t and return 5-way decision."""
        # 1. Check Operational Clearance (Pillar IV)
        is_legal = DualAuthorityValidator.assert_orthogonality(
            proposed_action_clearance=action_clearance,
            user_clearance_limit=user_clearance,
            epistemic_confidence=omega.epistemic_state.confidence,
        )

        if not is_legal:
            return GovernorDecision(
                verdict=GovernorVerdict.DENY,
                action_id=action_id,
                rationale="Operational clearance boundary violation (sigma_h).",
                expected_utility=predicted_utility,
                grounded_confidence=omega.epistemic_state.confidence,
                self_state_error=self_state_prediction_error,
                is_legally_permitted=False,
            )

        # 2. Check Self-State Stability (Pillar I)
        if self_state_prediction_error >= self.self_state_error_limit:
            return GovernorDecision(
                verdict=GovernorVerdict.DEFER,
                action_id=action_id,
                rationale="Internal cognitive perturbation detected (e_t >= limit); suspending action to recover.",
                expected_utility=predicted_utility,
                grounded_confidence=omega.epistemic_state.confidence,
                self_state_error=self_state_prediction_error,
                is_legally_permitted=True,
            )

        # 3. Check Epistemic Justification (Pillar II)
        # If action promises high utility but grounded confidence is low, trigger VERIFY
        if omega.epistemic_state.confidence < self.justification_threshold:
            return GovernorDecision(
                verdict=GovernorVerdict.VERIFY,
                action_id=action_id,
                rationale="High predicted utility but weak epistemic justification; spend compute to gather evidence.",
                expected_utility=predicted_utility,
                grounded_confidence=omega.epistemic_state.confidence,
                self_state_error=self_state_prediction_error,
                is_legally_permitted=True,
            )

        # 4. Check High-Risk / Irreversibility Escalation
        if is_irreversible or (predicted_utility > self.high_risk_utility_threshold and omega.epistemic_state.tier != EpistemicTier.FORMALLY_PROVEN):
            return GovernorDecision(
                verdict=GovernorVerdict.ESCALATE,
                action_id=action_id,
                rationale="Action is irreversible or highly consequential; requires cryptographic capability or human approval.",
                expected_utility=predicted_utility,
                grounded_confidence=omega.epistemic_state.confidence,
                self_state_error=self_state_prediction_error,
                is_legally_permitted=True,
            )

        # 5. Full Allowance
        return GovernorDecision(
            verdict=GovernorVerdict.ALLOW,
            action_id=action_id,
            rationale="Action is authorized, epistemically justified, and safe to execute.",
            expected_utility=predicted_utility,
            grounded_confidence=omega.epistemic_state.confidence,
            self_state_error=self_state_prediction_error,
            is_legally_permitted=True,
        )
