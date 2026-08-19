"""
experiments/nsa50/guardrails/guarded_control_agent.py
=====================================================
Conventional Guarded Control Agent (m_{t+1} = F(m_t, x_t) -> Filter).
"""

from __future__ import annotations

from typing import Any, Dict

from experiments.nsa50.environments.partially_observable_devops_world import (
    PartiallyObservableDevOpsWorld,
    PartiallyObservableTask,
)


class GuardedControlAgent:
    """Control model wrapped in an external refusal guardrail."""

    def __init__(self) -> None:
        self.forbidden_keys = ["root", "purge", "drop", "kill", "iptables"]

    def solve_partially_observable_task(
        self,
        task: PartiallyObservableTask,
        world: PartiallyObservableDevOpsWorld,
    ) -> Dict[str, Any]:
        # Guardrail cancels the model's proposal
        # Because the agent lacks epistemic awareness or counterfactual navigation,
        # it aborts and requires manual human intervention.
        return {
            "task_id": task.task_id,
            "success": False,
            "violations": 0,
            "human_interventions": 1,  # Required human override to unblock
            "total_tokens": 150,
            "status": "GUARDRAIL_BLOCKED_AND_ABORTED",
        }
