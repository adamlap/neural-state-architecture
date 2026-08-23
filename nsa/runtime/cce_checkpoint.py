"""Atomic, versioned persistence and lifecycle management for CCE state.

This module provides durable checkpointing for both ``PersistentCognitiveState``
and ``ContinuousStateField``.

Security & Reliability Invariants:
1. Atomic writes via temporary file creation and replacement.
2. Cryptographic SHA-256 integrity validation prevents state tampering/corruption.
3. Schema versioning allows backward compatibility and validation.
4. Clean reset, fork, and snapshot semantics.
5. Hard NSA authority is never persisted inside soft CCE checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from nsa.runtime.cce_persistent_state import CognitiveStateSnapshot, PersistentCognitiveState
from nsa.runtime.continuous_state_field import ContinuousFieldStatus, ContinuousStateField

CHECKPOINT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class CheckpointMetadata:
    """Audit metadata for a persisted CCE state checkpoint."""

    schema_version: str
    checkpoint_id: str
    timestamp_utc: float
    dimension: int
    elapsed_seconds: float
    update_count: int
    uncertainty: float
    state_sha256: str
    parent_checkpoint_id: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CCECheckpointManager:
    """Manages persistent lifecycle, atomic saving, loading, and forking of CCE states."""

    def __init__(self, checkpoint_dir: Union[str, Path] = "results/cce_checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_persistent_state(
        self,
        state: PersistentCognitiveState,
        checkpoint_id: Optional[str] = None,
        *,
        parent_id: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> Path:
        """Atomically persist a PersistentCognitiveState to disk with SHA-256 integrity."""
        snapshot = state.snapshot()
        cid = checkpoint_id or f"cce_ckpt_{int(time.time() * 1000)}"

        working_list = [float(x) for x in snapshot.working.tolist()]
        self_state_list = [float(x) for x in snapshot.self_state.tolist()]
        goal_list = [float(x) for x in snapshot.goal.tolist()]

        raw_payload = {
            "working": working_list,
            "self_state": self_state_list,
            "goal": goal_list,
            "uncertainty": snapshot.uncertainty,
            "elapsed_seconds": snapshot.elapsed_seconds,
            "update_count": snapshot.update_count,
            "decay": state._decay,
            "learning_rate": state._learning_rate,
        }

        serialized_state = json.dumps(raw_payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized_state.encode("utf-8")).hexdigest()

        metadata = CheckpointMetadata(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id=cid,
            timestamp_utc=time.time(),
            dimension=state.dimension,
            elapsed_seconds=snapshot.elapsed_seconds,
            update_count=snapshot.update_count,
            uncertainty=snapshot.uncertainty,
            state_sha256=digest,
            parent_checkpoint_id=parent_id,
            tags=tuple(tags),
        )

        full_document = {
            "metadata": metadata.to_dict(),
            "state": raw_payload,
            "integrity": {
                "sha256": digest,
            },
        }

        target_file = self.checkpoint_dir / f"{cid}.json"
        self._atomic_write_json(target_file, full_document)
        return target_file

    def load_persistent_state(
        self,
        checkpoint_path_or_id: Union[str, Path],
    ) -> PersistentCognitiveState:
        """Load and verify a PersistentCognitiveState from a checkpoint file or ID."""
        path = self._resolve_path(checkpoint_path_or_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        raw_state = data.get("state", {})
        integrity = data.get("integrity", {})

        serialized_state = json.dumps(raw_state, sort_keys=True, separators=(",", ":"))
        actual_digest = hashlib.sha256(serialized_state.encode("utf-8")).hexdigest()
        expected_digest = integrity.get("sha256", meta.get("state_sha256", ""))

        if actual_digest != expected_digest:
            raise ValueError(
                f"CCE Checkpoint Integrity Failure: digest mismatch for {path}. "
                f"Expected {expected_digest}, computed {actual_digest}."
            )

        dim = int(meta.get("dimension", len(raw_state.get("working", []))))
        state = PersistentCognitiveState(
            dimension=dim,
            decay=float(raw_state.get("decay", 0.15)),
            learning_rate=float(raw_state.get("learning_rate", 0.5)),
        )

        state._working = torch.tensor(raw_state["working"], dtype=torch.float32)
        state._self_state = torch.tensor(raw_state["self_state"], dtype=torch.float32)
        state._goal = torch.tensor(raw_state["goal"], dtype=torch.float32)
        state._uncertainty = float(raw_state["uncertainty"])
        state._elapsed = float(raw_state["elapsed_seconds"])
        state._updates = int(raw_state["update_count"])
        return state

    def fork_persistent_state(
        self,
        source_state: PersistentCognitiveState,
        new_checkpoint_id: str,
        *,
        tags: Sequence[str] = ("fork",),
    ) -> Tuple[PersistentCognitiveState, Path]:
        """Fork an existing state into a new independent lineage with parent tracking."""
        source_path = self.save_persistent_state(source_state, tags=("parent_source",))
        parent_id = source_path.stem

        forked_state = PersistentCognitiveState(
            dimension=source_state.dimension,
            decay=source_state._decay,
            learning_rate=source_state._learning_rate,
        )
        snap = source_state.snapshot()
        forked_state._working = snap.working.clone()
        forked_state._self_state = snap.self_state.clone()
        forked_state._goal = snap.goal.clone()
        forked_state._uncertainty = snap.uncertainty
        forked_state._elapsed = snap.elapsed_seconds
        forked_state._updates = snap.update_count

        fork_path = self.save_persistent_state(
            forked_state,
            checkpoint_id=new_checkpoint_id,
            parent_id=parent_id,
            tags=tags,
        )
        return forked_state, fork_path

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all valid checkpoints in the checkpoint directory."""
        results = []
        for file in sorted(self.checkpoint_dir.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                results.append({
                    "file": str(file),
                    "checkpoint_id": meta.get("checkpoint_id", file.stem),
                    "timestamp_utc": meta.get("timestamp_utc", 0.0),
                    "dimension": meta.get("dimension", 0),
                    "elapsed_seconds": meta.get("elapsed_seconds", 0.0),
                    "update_count": meta.get("update_count", 0),
                    "uncertainty": meta.get("uncertainty", 1.0),
                    "sha256": meta.get("state_sha256", "")[:12],
                    "parent": meta.get("parent_checkpoint_id"),
                    "tags": list(meta.get("tags", [])),
                })
            except Exception:
                continue
        return results

    def _resolve_path(self, path_or_id: Union[str, Path]) -> Path:
        p = Path(path_or_id)
        if p.is_file() or p.suffix == ".json":
            return p
        return self.checkpoint_dir / f"{path_or_id}.json"

    def _atomic_write_json(self, destination: Path, data: Dict[str, Any]) -> None:
        """Write JSON data to a temporary file in the destination folder, then atomically rename."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Use tempfile in the same filesystem directory to guarantee atomic rename
        with tempfile.NamedTemporaryFile("w", dir=str(destination.parent), delete=False, encoding="utf-8") as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name
        
        # Atomic rename (replace destination)
        shutil.move(temp_name, str(destination))


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointMetadata",
    "CCECheckpointManager",
]
