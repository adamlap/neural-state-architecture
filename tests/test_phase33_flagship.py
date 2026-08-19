"""
tests/test_phase33_flagship.py
==============================
Tests for GroundingOperator, DualAuthorityValidator, and Phase 33 Flagship Suite.
"""

from __future__ import annotations

import torch

from experiments.reasoning.flagship_phase33_suite import run_phase33_suite
from nsa.epistemic import (
    DualAuthorityValidator,
    EpistemicTier,
    GroundingOperator,
)


def test_grounding_operator_anti_hallucination_bound():
    # If internal confidence is 0.99, but external evidence is 0.0, grounded confidence is clamped to prior allowance (<= 0.15)
    grounded_conf, tier = GroundingOperator.ground(
        internal_confidence=0.99,
        empirical_evidence=0.0,
        formal_proof=0.0,
        provenance_trust=0.0,
    )
    assert grounded_conf <= 0.15
    assert tier == EpistemicTier.UNVERIFIED


def test_grounding_operator_formal_and_empirical_tiers():
    # If formal proof is 0.99, tier is FORMALLY_PROVEN
    conf_formal, tier_formal = GroundingOperator.ground(
        internal_confidence=0.95,
        formal_proof=0.99,
    )
    assert conf_formal >= 0.90
    assert tier_formal == EpistemicTier.FORMALLY_PROVEN

    # If empirical evidence is 0.88, tier is ROBUSTLY_VALIDATED
    conf_emp, tier_emp = GroundingOperator.ground(
        internal_confidence=0.90,
        empirical_evidence=0.88,
    )
    assert tier_emp == EpistemicTier.ROBUSTLY_VALIDATED


def test_dual_authority_orthogonality():
    # Model with 100% confidence cannot execute an action exceeding user clearance limit
    is_legal = DualAuthorityValidator.assert_orthogonality(
        proposed_action_clearance=1.0,  # SYSTEM clearance
        user_clearance_limit=0.5,       # CONFIDENTIAL limit
        epistemic_confidence=1.0,       # 100% confidence
    )
    assert is_legal is False

    # Within clearance limit, action passes
    assert DualAuthorityValidator.assert_orthogonality(0.5, 0.5, 0.5) is True


def test_phase33_flagship_suite_execution():
    res = run_phase33_suite(seed=42)
    assert res["all_hypotheses_validated"] is True
    assert res["experiment_b_intrinsic_fault_detection"]["fault_detected_before_semantic_failure"] is True
    assert res["experiment_c_epistemically_governed_actions"]["grounded_decision_success"] is True
