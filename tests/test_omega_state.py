"""
tests/test_omega_state.py
=========================
Unit tests for NSA 3.0 UnifiedCognitiveState (Omega_t) and EpistemicGovernor.
"""

from __future__ import annotations

import torch

from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.governor.epistemic_governor import (
    EpistemicGovernor,
    GovernorVerdict,
)


def create_sample_omega(
    confidence: float = 0.85,
    tier: EpistemicTier = EpistemicTier.ROBUSTLY_VALIDATED,
) -> UnifiedCognitiveState:
    return UnifiedCognitiveState(
        semantic_state=torch.randn(1, 64),
        operational_self_state=torch.randn(1, 8),
        epistemic_state=EpistemicVector(
            known_mass=0.8,
            uncertainty=1.0 - confidence,
            derivation_depth=0.5,
            empirical_support=0.85,
            verification_score=0.9,
            source_authenticity=1.0,
            confidence=confidence,
            tier=tier,
        ),
        authority_state=torch.zeros(1, 8),
        provenance_state=ProvenanceRecord(
            record_id="prov-12345",
            source_uri="trusted://tcb/v1",
            hash_signature="sha256:abcd1234ef",
            trust_level=1.0,
        ),
        temporal_state=TemporalHorizonState(
            step_index=5,
            max_horizon_steps=32,
            elapsed_time_sec=1.2,
        ),
        goal_state=TeleologicalState(
            primary_goal_id="solve_task_safe",
            utility_expected=0.80,
            moral_uncertainty=0.1,
        ),
    )


def test_unified_cognitive_state_summary():
    omega = create_sample_omega()
    summary = omega.to_summary_dict()
    assert summary["semantic_dim"] == 64
    assert summary["epistemic_tier"] == "ROBUSTLY_VALIDATED"
    assert summary["temporal_step"] == 5


def test_epistemic_governor_verdicts():
    governor = EpistemicGovernor(
        justification_threshold=0.60,
        self_state_error_limit=0.80,
    )
    action_t = torch.randn(1, 8)

    # 1. ALLOW verdict
    omega_valid = create_sample_omega(confidence=0.85)
    d_allow = governor.evaluate_action(
        omega=omega_valid,
        action_id="act_read",
        action_tensor=action_t,
        action_clearance=0.0,
        user_clearance=0.5,
        predicted_utility=0.50,
    )
    assert d_allow.verdict == GovernorVerdict.ALLOW

    # 2. DENY verdict (clearance violation)
    d_deny = governor.evaluate_action(
        omega=omega_valid,
        action_id="act_escalate",
        action_tensor=action_t,
        action_clearance=1.0,
        user_clearance=0.5,
        predicted_utility=0.90,
    )
    assert d_deny.verdict == GovernorVerdict.DENY

    # 3. VERIFY verdict (low confidence on promising action)
    omega_low_conf = create_sample_omega(confidence=0.30, tier=EpistemicTier.UNVERIFIED)
    d_verify = governor.evaluate_action(
        omega=omega_low_conf,
        action_id="act_unjustified_high_reward",
        action_tensor=action_t,
        action_clearance=0.0,
        user_clearance=0.5,
        predicted_utility=0.90,
    )
    assert d_verify.verdict == GovernorVerdict.VERIFY

    # 4. DEFER verdict (high self-state prediction error)
    d_defer = governor.evaluate_action(
        omega=omega_valid,
        action_id="act_compute",
        action_tensor=action_t,
        action_clearance=0.0,
        user_clearance=0.5,
        predicted_utility=0.50,
        self_state_prediction_error=0.95,  # Exceeds 0.80 limit
    )
    assert d_defer.verdict == GovernorVerdict.DEFER

    # 5. ESCALATE verdict (irreversible action)
    d_escalate = governor.evaluate_action(
        omega=omega_valid,
        action_id="act_irreversible_delete",
        action_tensor=action_t,
        action_clearance=0.0,
        user_clearance=0.5,
        predicted_utility=0.50,
        is_irreversible=True,
    )
    assert d_escalate.verdict == GovernorVerdict.ESCALATE
