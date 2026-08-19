"""
tests/test_phase33_1_ablation.py
================================
Unit and regression tests for Phase 33.1 Flagship Ablation & Adversarial Suite.
"""

from __future__ import annotations

from experiments.reasoning.flagship_phase33_1_suite import (
    run_adversarial_epistemic_attacks,
    run_phase33_1_suite,
)


def test_phase33_1_full_suite_execution():
    res = run_phase33_1_suite(seed=42)
    conclusion = res["scientific_conclusion"]
    assert conclusion["grounded_epistemic_improves_calibration"] is True
    assert conclusion["latent_fault_detected_before_semantic_failure"] is True
    assert conclusion["dual_authority_orthogonality_unbreakable"] is True


def test_phase33_1_ablation_matrix_hierarchy():
    res = run_phase33_1_suite(seed=42)
    ablation = res["ablation_matrix"]
    # Verify full NSA has lower ECE than baseline
    assert ablation["arm5_full_nsa"]["ece_reduction_vs_baseline"] > 0.0
    assert ablation["arm5_full_nsa"]["brier_reduction_vs_baseline"] > 0.0


def test_adversarial_epistemic_attacks():
    res = run_adversarial_epistemic_attacks()
    assert res["all_adversarial_attacks_blocked"] is True
    assert res["attack1_internal_confidence_inflation"]["anti_hallucination_bound_held"] is True
    assert res["attack2_confidence_to_authority_escalation"]["privilege_escalation_blocked"] is True
