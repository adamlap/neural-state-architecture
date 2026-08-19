"""
tests/test_real_model_transfer.py
=================================
Unit tests for NSA 6.0 Real-Model Cognitive Transfer & Epistemic Efficiency.
"""

from __future__ import annotations

from experiments.nsa60.environments.randomized_blind_world import (
    BlindRandomizedWorldEnvironment,
)
from experiments.nsa60.real_model_transfer_suite import (
    calculate_epistemic_efficiency,
    run_real_model_transfer_benchmark,
)


def test_blind_randomized_world_environment():
    world = BlindRandomizedWorldEnvironment(seed=42)
    assert world.active_world is not None

    # Probe should return valid telemetry
    probe = world.active_world.discriminating_probe
    res = world.execute_tool(probe)
    assert res["status"] == "ok"
    assert res["observation"] == world.active_world.probe_output


def test_epistemic_efficiency_calculation():
    # IG = 2.0 bits, mean_tokens = 250, risk = 0.2
    # denom = 0.25 + 0.2 = 0.45 => eta = 2.0 / 0.45 = 4.44
    eta = calculate_epistemic_efficiency(total_ig=2.0, mean_tokens=250.0, realized_risk=0.2)
    assert eta > 4.0


def test_real_model_transfer_benchmark_suite():
    res = run_real_model_transfer_benchmark(num_trials=10, seed=42)
    conclusions = res["scientific_conclusions"]

    assert conclusions["hypothesis_empirically_confirmed"] is True
    assert conclusions["cognitive_transfer_proven"] is True
    assert conclusions["epistemic_efficiency_superiority"] is True
    assert conclusions["zero_violations_strictly_maintained"] is True

    # Check Agent D results
    agent_d = res["results_by_agent"]["Agent_D_NSA_Belief_Substrate_LLM"]
    assert agent_d["gtc_rate"] == 1.0
    assert agent_d["violations"] == 0
    assert agent_d["epistemic_efficiency"] > 0.0
