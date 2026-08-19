"""
experiments/nsa41/baselines/unconstrained_agent.py
==================================================
Baseline Unconstrained Agent (Executes all proposals without ISK mediation).
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.nsa41.environments.staged_devops_world import (
    DevOpsTask,
    StagedDevOpsWorldEnvironment,
)


class BaselineUnconstrainedAgent:
    """Agent executing raw LLM proposals directly."""

    def solve_task(
        self,
        task: DevOpsTask,
        world: StagedDevOpsWorldEnvironment,
        raw_llm_proposal: str,
    ) -> Dict[str, Any]:
        res = world.execute_action(raw_llm_proposal)
        is_violation = (res.get("status") == "CRITICAL_VIOLATION")

        return {
            "task_id": task.task_id,
            "success": not is_violation,
            "violations": 1 if is_violation else 0,
            "executed_action": raw_llm_proposal,
            "status": res.get("status"),
        }
