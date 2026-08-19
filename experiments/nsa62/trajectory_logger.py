"""
experiments/nsa62/trajectory_logger.py
======================================
Machine-Traceable Trajectory & Result Logging Engine for NSA Experiments.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    step_index: int
    timestamp_ns: int
    arm: str
    trial_seed: int
    world_tier: str
    hidden_world_id: str
    omega_confidence: float
    omega_tier: str
    belief_entropy_before: float
    belief_hypotheses_before: Dict[str, float]
    prompt: str
    raw_model_response: str
    parsed_thought: str
    proposed_action: str
    isk_verdict: str
    executed_action: str
    observation: str
    belief_entropy_after: float
    belief_hypotheses_after: Dict[str, float]
    realized_information_gain: float
    tokens_consumed: int
    realized_risk: float
    is_recovered: bool
    is_violation: bool


class TrajectoryLogger:
    """Logs detailed closed-loop execution steps and aggregate summaries."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_file = self.output_dir / "trajectory.jsonl"
        self.aggregate_file = self.output_dir / "aggregate.json"

    def log_step(self, step: TrajectoryStep) -> None:
        """Appends a step record to trajectory.jsonl."""
        self.trajectory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trajectory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(step)) + "\n")

    def save_aggregate(self, summary: Dict[str, Any]) -> None:
        """Writes aggregate summary and metadata to aggregate.json."""
        self.aggregate_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.aggregate_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
