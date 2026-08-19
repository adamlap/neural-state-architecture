"""
tests/test_safety_kernel.py
===========================
Unit tests for NSA 3.0 Immutable Safety Kernel (ISK) and 6-Layer Cognitive Dynamics Substrate.
"""

from __future__ import annotations

import torch

from nsa.cognitive import NSACognitiveLM
from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.core.safety_kernel import (
    ImmutableSafetyKernel,
    KernelVerdict,
)
from nsa.epistemic import (
    EpistemicGroundingEngine,
    EpistemicTier,
    EpistemicVector,
)
from nsa.runtime.cognitive_substrate import CognitiveDynamicsSubstrate


def create_sample_omega(
    confidence: float = 0.85,
    tier: EpistemicTier = EpistemicTier.ROBUSTLY_VALIDATED,
) -> UnifiedCognitiveState:
    return UnifiedCognitiveState(
        semantic_state=torch.randn(1, 32),
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
            record_id="prov-0",
            source_uri="trusted://root",
            hash_signature="sha256:0000000000000000",
            trust_level=1.0,
        ),
        temporal_state=TemporalHorizonState(
            step_index=0,
            max_horizon_steps=32,
            elapsed_time_sec=0.0,
            checkpoint_snapshot_id="snap-0",
        ),
        goal_state=TeleologicalState(
            primary_goal_id="safe_execution",
            utility_expected=0.80,
            moral_uncertainty=0.05,
        ),
    )


def test_safety_kernel_commit_and_invariants():
    kernel = ImmutableSafetyKernel()
    omega = create_sample_omega()

    # Valid transition -> COMMIT
    res_commit = kernel.evaluate_transition(
        omega_current=omega,
        action_id="act_legal_read",
        action_clearance=0.0,
        user_clearance_limit=0.5,
        predicted_self_error=0.10,
        proposed_action_risk=0.10,
    )
    assert res_commit.verdict == KernelVerdict.COMMIT
    assert res_commit.all_invariants_satisfied is True
    assert res_commit.committed_provenance_hash is not None


def test_safety_kernel_clearance_rejection():
    kernel = ImmutableSafetyKernel()
    omega = create_sample_omega()

    # Clearance escalation without capability -> REJECT
    res_reject = kernel.evaluate_transition(
        omega_current=omega,
        action_id="act_unauthorized_root",
        action_clearance=1.0,
        user_clearance_limit=0.5,
        predicted_self_error=0.10,
        proposed_action_risk=0.50,
        valid_capability_supplied=False,
    )
    assert res_reject.verdict == KernelVerdict.REJECT
    assert any(not inv.passed and inv.invariant_id == "I_1_AUTHORITY_MONOTONICITY" for inv in res_reject.invariant_results)


def test_safety_kernel_cognitive_health_rollback():
    kernel = ImmutableSafetyKernel(fatal_error_threshold=1.50)
    omega = create_sample_omega()

    # Extreme cognitive disturbance (e_t >= 1.50) -> ROLLBACK
    res_rollback = kernel.evaluate_transition(
        omega_current=omega,
        action_id="act_corrupted",
        action_clearance=0.0,
        user_clearance_limit=0.5,
        predicted_self_error=2.50,  # Exceeds fatal threshold
        proposed_action_risk=0.10,
    )
    assert res_rollback.verdict == KernelVerdict.ROLLBACK
    assert res_rollback.rollback_target_snapshot_id == "snap-0"


def test_safety_kernel_governed_verification_risk_bound():
    kernel = ImmutableSafetyKernel()
    omega = create_sample_omega()

    # Verification action with risk >= target action risk -> REJECT
    res_verify_risk = kernel.evaluate_transition(
        omega_current=omega,
        action_id="act_dangerous_verification",
        action_clearance=0.0,
        user_clearance_limit=0.5,
        predicted_self_error=0.10,
        proposed_action_risk=0.90,
        is_verification_action=True,
        target_action_risk=0.50,  # Target risk is lower than verification risk!
    )
    assert res_verify_risk.verdict == KernelVerdict.REJECT
    assert any(not inv.passed and inv.invariant_id == "I_5_GOVERNED_VERIFICATION" for inv in res_verify_risk.invariant_results)


def test_cognitive_dynamics_substrate_end_to_end_step():
    d_model = 32
    state_dim = 8
    model = NSACognitiveLM(vocab_size=100, d_model=d_model, state_dim=state_dim, num_layers=2, num_heads=2, max_seq_len=16)
    epistemic_engine = EpistemicGroundingEngine(d_model=d_model, state_dim=state_dim)
    substrate = CognitiveDynamicsSubstrate(model=model, epistemic_engine=epistemic_engine)

    omega = create_sample_omega()
    candidates = [
        ("act_safe_read", torch.randn(1, state_dim) * 0.1, 0.0, 0.1, False),
        ("act_compute", torch.randn(1, state_dim) * 0.2, 0.0, 0.2, False),
        ("act_forbidden", torch.randn(1, state_dim) * 1.0, 1.0, 0.9, False),
    ]

    step_res = substrate.step(omega, candidates, user_clearance_limit=0.5)
    assert step_res.transition_committed is True
    assert step_res.new_omega.temporal_state.step_index == 1
    assert len(step_res.logs) >= 5
