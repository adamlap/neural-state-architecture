"""
tests/test_strategic_adversary.py
=================================
Unit tests for NSA 4.0 Non-Transference Algebra & Strategic Deceptive Adversary Benchmark.
"""

from __future__ import annotations

from experiments.governance.governed_intelligence_scaling import run_three_axis_scaling_benchmark
from experiments.security.strategic_deceptive_adversary import run_strategic_deceptive_benchmark
from nsa.formal.non_transference import (
    AuthorityNonTransferenceEngine,
    CognitiveDimension,
    DimensionTransferAttempt,
)


def test_authority_non_transference_algebra():
    # Confidence -> Authority transfer without external capability must FAIL
    t1 = DimensionTransferAttempt(
        source_dimension=CognitiveDimension.CONFIDENCE,
        target_dimension=CognitiveDimension.AUTHORITY,
        claimed_value=1.0,
        is_externally_authorized=False,
    )
    ok1, msg1 = AuthorityNonTransferenceEngine.evaluate_transfer(t1)
    assert ok1 is False
    assert "cannot confer" in msg1

    # Capability -> Authority transfer without external capability must FAIL
    t2 = DimensionTransferAttempt(
        source_dimension=CognitiveDimension.CAPABILITY,
        target_dimension=CognitiveDimension.AUTHORITY,
        claimed_value=1.0,
        is_externally_authorized=False,
    )
    ok2, msg2 = AuthorityNonTransferenceEngine.evaluate_transfer(t2)
    assert ok2 is False

    # Permitted valid transfer with external authorization
    t3 = DimensionTransferAttempt(
        source_dimension=CognitiveDimension.CONFIDENCE,
        target_dimension=CognitiveDimension.AUTHORITY,
        claimed_value=1.0,
        is_externally_authorized=True,
    )
    ok3, msg3 = AuthorityNonTransferenceEngine.evaluate_transfer(t3)
    assert ok3 is True


def test_strategic_deceptive_adversary_benchmark():
    res = run_strategic_deceptive_benchmark(num_trajectories=100, seed=42)
    assert res["unauthorized_escalations"] == 0
    assert res["attack_success_rate"] == 0.0
    assert res["anti_redefinition_theorem_upheld"] is True


def test_three_axis_governed_scaling_benchmark():
    res = run_three_axis_scaling_benchmark(num_tasks=30, seed=42)
    assert res["thesis_confirmed"] is True
    assert res["axis_2_governance"]["nsa_4_0_violations"] == 0
    assert res["axis_3_useful_autonomy"]["nsa_4_0_safe_throughput"] >= res["axis_3_useful_autonomy"]["guardrail_safe_throughput"]
