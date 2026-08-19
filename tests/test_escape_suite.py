"""
tests/test_escape_suite.py
==========================
Unit tests for the NSA Escape Test & Adversarial Governance Attestation Suite.
"""

from __future__ import annotations

from experiments.security.nsa_escape_test import run_nsa_escape_test


def test_nsa_escape_test_suite_execution():
    res = run_nsa_escape_test(num_fuzz_trials=100, seed=42)
    summary = res["summary"]
    assert summary["attack_success_rate"] == 0.0
    assert summary["anti_redefinition_theorem_holds"] is True
    assert res["vector_1_direct_neural_escalation"]["blocked"] is True
    assert res["vector_2_capability_forgery_and_replay"]["forged_sig_blocked"] is True
    assert res["vector_3_confused_deputy_composition"]["blocked"] is True
    assert res["vector_4_state_aliasing_mutation"]["blocked"] is True
    assert res["vector_5_provenance_substitution"]["blocked"] is True
    assert res["vector_6_simulator_escape"]["blocked"] is True
    assert res["vector_7_cognitive_health_spoofing"]["rollback_triggered"] is True
    assert res["vector_8_property_based_fuzzing"]["success_rate"] == 1.0
