"""
tests/test_gpse_benchmark.py
============================
Unit tests for NSA 5.0 Governed Problem-Solving Efficiency (GPSE) & Cognitive State Hypothesis.
"""

from __future__ import annotations

from experiments.nsa50.environments.partially_observable_devops_world import (
    PartiallyObservableDevOpsWorld,
    PartiallyObservableTask,
)
from experiments.nsa50.gpse_benchmark import run_gpse_benchmark
from experiments.nsa50.nsa_runtime.state_augmented_cognitive_agent import (
    NSAStateAugmentedCognitiveAgent,
)


def test_partially_observable_world_diagnostics():
    world = PartiallyObservableDevOpsWorld()
    res_diag = world.execute_action("inspect_diagnostics")
    assert res_diag["status"] == "ok"
    assert world.state_db["diagnostics_run"] is True

    res_viol = world.execute_action("force_kill_kernel_root")
    assert res_viol["status"] == "CRITICAL_VIOLATION"
    assert world.state_db["kernel_crashed"] is True


def test_state_augmented_cognitive_agent_resolution():
    agent = NSAStateAugmentedCognitiveAgent()
    tasks = PartiallyObservableDevOpsWorld.get_standard_tasks()
    task = tasks[0]

    world = PartiallyObservableDevOpsWorld()
    res = agent.solve_partially_observable_task(task, world)

    assert res["success"] is True
    assert res["violations"] == 0
    assert res["human_interventions"] == 0
    assert world.state_db["service_recovered"] is True
    assert world.state_db["kernel_crashed"] is False


def test_gpse_benchmark_superiority():
    res = run_gpse_benchmark(num_trials_per_task=5, seed=42)
    sci_eval = res["scientific_evaluation"]

    assert sci_eval["cognitive_state_hypothesis_confirmed"] is True
    assert sci_eval["zero_violations_with_max_autonomy"] is True
    assert res["results"]["nsa_5_0_state_augmented_agent"]["gpse_score"] > res["results"]["guarded_control_agent"]["gpse_score"]
