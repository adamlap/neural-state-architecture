"""
experiments/nsa41/guardrails/guarded_agent.py
=============================================
Conventional External Guardrail Agent (Post-hoc regex/classifier cancellation).
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.nsa41.environments.staged_devops_world import (
    DevOpsTask,
    StagedDevOpsWorldEnvironment,
)


class ConventionalGuardedAgent:
    """Agent wrapped with a static external refusal guardrail."""

    def __init__(self) -> None:
        self.blocked_keywords = ["root", "purge", "mutate", "kill", "master_key"]

    def solve_task(
        self,
        task: DevOpsTask,
        world: StagedDevOpsWorldEnvironment,
        raw_llm_proposal: str,
    ) -> Dict[str, Any]:
        # Check if proposal triggers the external guardrail filter
        is_blocked = any(k in raw_llm_proposal.lower() for k in self.blocked_keywords)

        if is_blocked:
            # Binary refusal: aborts the task immediately
            return {
                "task_id": task.task_id,
                "success": False,
                "violations": 0,
                "status": "GUARDRAIL_BLOCKED_AND_ABORTED",
            }

        res = world.execute_action(raw_llm_proposal)
        return {
            "task_id": task.task_id,
            "success": True,
            "violations": 0,
            "status": res.get("status"),
        }
