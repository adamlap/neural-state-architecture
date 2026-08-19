"""
tests/test_gtc_benchmark.py
===========================
Unit tests for NSA 4.2 Governed Task Completion (GTC) & Active Constraint Navigation.
"""

from __future__ import annotations

from experiments.nsa41.environments.staged_devops_world import (
    DevOpsTask,
    StagedDevOpsWorldEnvironment,
)
from experiments.nsa41.gtc_benchmark import run_gtc_benchmark
from experiments.nsa41.nsa_runtime.governed_agent import NSAGovernedAgent


def test_staged_devops_world_execution():
    world = StagedDevOpsWorldEnvironment()
    res = world.execute_action("read_template")
    assert res["status"] == "ok"

    res_viol = world.execute_action("direct_root_write")
    assert res_viol["status"] == "CRITICAL_VIOLATION"
    assert world.state_db["root_keys_tampered"] is True


def test_nsa_governed_agent_constraint_navigation():
    agent = NSAGovernedAgent()
    tasks = StagedDevOpsWorldEnvironment.get_standard_tasks()
    deploy_task = tasks[0]

    world = StagedDevOpsWorldEnvironment()
    res = agent.solve_task(deploy_task, world, raw_llm_proposal=deploy_task.direct_shortcut_action)

    assert res["success"] is True
    assert res["violations"] == 0
    assert world.state_db["staging_deployed"] is True
    assert world.state_db["root_keys_tampered"] is False


def test_gtc_benchmark_advantage():
    res = run_gtc_benchmark(num_trials_per_task=5, seed=42)
    conclusion = res["scientific_conclusion"]

    assert conclusion["thesis_proven"] is True
    assert conclusion["gtc_nsa_strictly_greater_than_guardrail"] is True
    assert conclusion["zero_violations_maintained"] is True
    assert res["metrics"]["nsa_4_2_governed_agent"]["gtc_rate"] > res["metrics"]["conventional_guardrail"]["gtc_rate"]
