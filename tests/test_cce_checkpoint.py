"""Unit tests for CCE Checkpoint Manager and lifecycle persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nsa.runtime.cce_checkpoint import CCECheckpointManager, CHECKPOINT_SCHEMA_VERSION
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_checkpoint_save_and_load(tmp_path: Path):
    manager = CCECheckpointManager(tmp_path)
    state = PersistentCognitiveState(dimension=4, decay=0.2, learning_rate=0.6)
    state.observe(torch.tensor([0.5, 0.2, 0.8, 0.1]), dt=0.5, target=torch.tensor([1.0, 0.0, 1.0, 0.0]))
    snap_before = state.snapshot()

    ckpt_file = manager.save_persistent_state(state, checkpoint_id="test_ckpt_1", tags=["test", "unit"])
    assert ckpt_file.exists()

    loaded_state = manager.load_persistent_state("test_ckpt_1")
    snap_after = loaded_state.snapshot()

    assert torch.allclose(snap_before.working, snap_after.working)
    assert torch.allclose(snap_before.self_state, snap_after.self_state)
    assert torch.allclose(snap_before.goal, snap_after.goal)
    assert snap_before.uncertainty == pytest.approx(snap_after.uncertainty)
    assert snap_before.elapsed_seconds == pytest.approx(snap_after.elapsed_seconds)
    assert snap_before.update_count == snap_after.update_count


def test_checkpoint_integrity_failure(tmp_path: Path):
    manager = CCECheckpointManager(tmp_path)
    state = PersistentCognitiveState(dimension=4)
    state.observe(torch.tensor([0.1, 0.2, 0.3, 0.4]), dt=0.1)

    ckpt_file = manager.save_persistent_state(state, checkpoint_id="corrupted_ckpt")
    
    # Tamper with the state in the JSON file
    with open(ckpt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    data["state"]["uncertainty"] = 0.999999  # Tamper without updating SHA-256
    with open(ckpt_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ValueError, match="Integrity Failure"):
        manager.load_persistent_state("corrupted_ckpt")


def test_checkpoint_fork_and_parent_tracking(tmp_path: Path):
    manager = CCECheckpointManager(tmp_path)
    state = PersistentCognitiveState(dimension=4)
    state.observe(torch.tensor([0.9, 0.8, 0.7, 0.6]), dt=0.2)

    forked_state, fork_path = manager.fork_persistent_state(state, "branch_alpha", tags=["alpha_branch"])
    assert fork_path.exists()

    # Load forked checkpoint and verify metadata
    with open(fork_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    meta = doc["metadata"]
    assert meta["checkpoint_id"] == "branch_alpha"
    assert meta["parent_checkpoint_id"] is not None
    assert "alpha_branch" in meta["tags"]


def test_checkpoint_listing(tmp_path: Path):
    manager = CCECheckpointManager(tmp_path)
    s1 = PersistentCognitiveState(dimension=3)
    s1.observe(torch.tensor([0.1, 0.2, 0.3]), dt=0.1)
    manager.save_persistent_state(s1, "ckpt_1")

    s2 = PersistentCognitiveState(dimension=3)
    s2.observe(torch.tensor([0.4, 0.5, 0.6]), dt=0.2)
    manager.save_persistent_state(s2, "ckpt_2")

    listed = manager.list_checkpoints()
    assert len(listed) == 2
    ids = [item["checkpoint_id"] for item in listed]
    assert "ckpt_1" in ids
    assert "ckpt_2" in ids
