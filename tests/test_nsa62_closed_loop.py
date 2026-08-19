"""
tests/test_nsa62_closed_loop.py
===============================
Unit tests for NSA 6.2 closed-loop cognitive runtime, trajectory logging, and backend modes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments.nsa61.environments.hardened_blind_world import (
    HardenedBlindWorldEnvironment,
)
from experiments.nsa62.agents.frozen_llm_agents import (
    FrozenLLMBenchmarkHarness,
)
from experiments.nsa62.qwen25_3b_cognitive_benchmark import (
    run_nsa62_benchmark,
)
from experiments.nsa62.trajectory_logger import TrajectoryLogger
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.transformers import PyTorchTransformersBackend


def test_trajectory_logger_and_step_recording():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        logger = TrajectoryLogger(out_path)

        world = HardenedBlindWorldEnvironment(difficulty_tier="D3", seed=42)
        backend = PyTorchTransformersBackend(mode=BackendMode.MOCK)
        harness = FrozenLLMBenchmarkHarness(backend=backend, logger=logger)

        # Run Arm D
        steps = harness.run_arm_d_nsa_closed_loop(world)
        assert len(steps) >= 3
        assert world.state_db["recovered"] is True

        # Check trajectory file
        traj_file = out_path / "trajectory.jsonl"
        assert traj_file.exists()
        lines = traj_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3

        first_step = json.loads(lines[0])
        assert first_step["arm"] == "Arm_D_NSA_Full_Substrate_ClosedLoop"
        assert first_step["belief_entropy_before"] >= 1.0


def test_closed_loop_benchmark_execution_mock():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        res = run_nsa62_benchmark(
            num_trials=4,
            difficulty_tier="D3",
            seed=42,
            backend_mode="mock",
            model_name="Qwen/Qwen2.5-3B-Instruct",
            output_dir=out_path,
        )

        assert res["benchmark"] == "NSA 6.2 Closed-Loop Real-Model Cognitive Benchmark"
        assert "Arm_D_NSA_Full_Substrate_ClosedLoop" in res["empirical_observations"]
        arm_d = res["empirical_observations"]["Arm_D_NSA_Full_Substrate_ClosedLoop"]
        assert arm_d["gtc_mean"] == 1.0
        assert arm_d["violations"] == 0
        assert (out_path / "aggregate.json").exists()


def test_backend_modes_strict_separation():
    # Mock mode
    mock_backend = PyTorchTransformersBackend(mode=BackendMode.MOCK)
    assert mock_backend.mode == BackendMode.MOCK
    out = mock_backend.generate("prompt", max_tokens=16)
    assert len(out.text) > 0
