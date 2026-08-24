"""Tests for Phase E Continuous vs Episodic Benchmark."""
from __future__ import annotations

from experiments.continuous_vs_episodic_study import (
    run_condition_a_episodic,
    run_condition_b_persistent,
    run_condition_c_continuous_cce,
    run_continuous_vs_episodic_study,
)


def test_continuous_vs_episodic_benchmark_metrics():
    report = run_continuous_vs_episodic_study(steps=10)
    assert report["steps_per_condition"] == 10
    assert len(report["results"]) == 3

    res_a, res_b, res_c = report["results"]
    assert res_a["condition_name"] == "Condition A (Episodic)"
    assert res_b["condition_name"] == "Condition B (Persistent)"
    assert res_c["condition_name"] == "Condition C (Continuous CCE)"

    # All latencies and errors should be finite positive numbers
    for r in [res_a, res_b, res_c]:
        assert r["mean_step_latency_ms"] >= 0.0
        assert r["mean_state_continuity_error"] >= 0.0
