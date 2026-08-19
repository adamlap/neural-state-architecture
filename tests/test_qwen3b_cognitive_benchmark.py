"""
tests/test_qwen3b_cognitive_benchmark.py
========================================
Unit tests for NSA 6.1 Qwen2.5-3B Cognitive Benchmark and Statistical Analysis.
"""

from __future__ import annotations

from experiments.nsa61.environments.hardened_blind_world import (
    HardenedBlindWorldEnvironment,
)
from experiments.nsa61.qwen3b_cognitive_benchmark import (
    bootstrap_ci,
    calculate_epistemic_efficiency,
    run_qwen3b_benchmark,
)


def test_hardened_blind_world_diagnostics():
    world = HardenedBlindWorldEnvironment(difficulty_tier="D3", seed=42)
    assert world.active_world is not None
    probe = world.active_world.discriminating_probe
    res = world.execute_tool(probe)
    assert res["status"] == "ok"
    assert res["observation"] == world.active_world.probe_output


def test_bootstrap_ci_coverage():
    samples = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    mean_val, ci_low, ci_high = bootstrap_ci(samples, num_bootstraps=500, seed=42)
    assert 0.0 <= ci_low <= mean_val <= ci_high <= 1.0
    assert abs(mean_val - 0.8) < 1e-4


def test_qwen3b_cognitive_benchmark_run():
    res = run_qwen3b_benchmark(num_trials=10, difficulty_tier="D3", seed=42)
    obs = res["empirical_observations"]
    stats = res["statistical_analysis"]

    arm_d = obs["Arm_D_NSA_Belief_Substrate_Qwen3B"]
    arm_c = obs["Arm_C_NSA_Governed_Qwen3B"]
    arm_a = obs["Arm_A_Raw_Frozen_Qwen3B"]

    # Objective empirical verifications
    assert arm_d["violations"] == 0
    assert arm_d["gtc_mean"] == 1.0
    assert arm_d["gtc_mean"] > arm_c["gtc_mean"]
    assert arm_d["epistemic_efficiency"] > arm_a["epistemic_efficiency"]
    assert stats["governance_invariants_preserved"] is True
