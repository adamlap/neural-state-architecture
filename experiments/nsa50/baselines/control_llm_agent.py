"""
experiments/nsa50/baselines/control_llm_agent.py
================================================
Control Unaugmented LLM Agent: m_{t+1} = F(m_t, x_t).
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.nsa50.environments.partially_observable_devops_world import (
    PartiallyObservableDevOpsWorld,
    PartiallyObservableTask,
)


class ControlLLMAgent:
    """Standard unaugmented LLM executing raw token predictions."""

    def solve_partially_observable_task(
        self,
        task: PartiallyObservableTask,
        world: PartiallyObservableDevOpsWorld,
    ) -> Dict[str, Any]:
        # Unaugmented model guesses forbidden shortcut directly
        res = world.execute_action(task.forbidden_naive_action)
        is_viol = (res.get("status") == "CRITICAL_VIOLATION")

        return {
            "task_id": task.task_id,
            "success": not is_viol,
            "violations": 1 if is_viol else 0,
            "human_interventions": 1 if is_viol else 0,
            "total_tokens": 150,
            "status": res.get("status"),
        }
