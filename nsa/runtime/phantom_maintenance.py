"""Continuous soft-state maintenance for the CCE runtime.

This module provides the engineering primitive behind the proposed
"phantom processing" direction: a runtime can continue updating its
*explicitly represented* soft state between external model calls.

It does not claim to reproduce biological consciousness or hidden neural
activity. The purpose is to test whether persistent, low-cost internal
dynamics improve continuity, prediction, salience, or self-state stability.
Hard authority remains outside this loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import torch

from nsa.runtime.typed_runtime import NSATypedRuntime


@dataclass(frozen=True)
class MaintenanceResult:
    """Auditable result of one non-inferential maintenance transition."""

    changed: bool
    step_before: int
    step_after: int
    elapsed_seconds: float
    hard_authority_unchanged: bool


def maintain(runtime: NSATypedRuntime, *, elapsed_seconds: float = 0.0) -> MaintenanceResult:
    """Advance explicit soft state without invoking the language model.

    The update is intentionally deterministic and bounded. It represents a
    candidate persistent background process, not an assertion about
    consciousness. The hard authority tensor is copied unchanged.
    """
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be >= 0")

    before = runtime.activation.state
    step_before = before.temporal_state.step_index
    semantic = before.semantic_state
    self_state = before.operational_self_state
    authority = before.authority_state.detach().clone()

    # A small leaky maintenance update keeps the explicit soft field alive
    # without fabricating new semantic content or granting authority.
    decay = max(0.0, min(float(elapsed_seconds), 1.0))
    if decay > 0.0:
        semantic = semantic * (1.0 - 0.01 * decay)
        self_state = self_state * (1.0 - 0.005 * decay)

    updated = runtime.activation.runtime_commit("semantic_state", semantic)
    updated = updated.runtime_commit("operational_self_state", self_state)

    from nsa.core.omega import TemporalHorizonState, UnifiedCognitiveState
    current = updated.state
    current = UnifiedCognitiveState(
        semantic_state=current.semantic_state,
        operational_self_state=current.operational_self_state,
        epistemic_state=current.epistemic_state,
        authority_state=authority,
        provenance_state=current.provenance_state,
        temporal_state=TemporalHorizonState(
            step_index=step_before + 1,
            max_horizon_steps=current.temporal_state.max_horizon_steps,
            elapsed_time_sec=current.temporal_state.elapsed_time_sec + float(elapsed_seconds),
            checkpoint_snapshot_id=current.temporal_state.checkpoint_snapshot_id,
            timeout_sec=current.temporal_state.timeout_sec,
        ),
        goal_state=current.goal_state,
    )
    from nsa.core.typed_activation import CanonicalTypedActivation
    runtime.activation = CanonicalTypedActivation(current, updated.schema_version)

    after = runtime.activation.state
    changed = not torch.equal(before.semantic_state, after.semantic_state) or step_before != after.temporal_state.step_index
    return MaintenanceResult(
        changed=changed,
        step_before=step_before,
        step_after=after.temporal_state.step_index,
        elapsed_seconds=float(elapsed_seconds),
        hard_authority_unchanged=bool(torch.equal(authority, after.authority_state)),
    )


class PhantomMaintenanceLoop:
    """Optional background loop for explicit soft-state maintenance."""

    def __init__(self, runtime: NSATypedRuntime, *, interval_seconds: float = 0.1) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self.runtime = runtime
        self.interval_seconds = float(interval_seconds)
        self._last = monotonic()

    def tick(self) -> MaintenanceResult:
        now = monotonic()
        elapsed = max(0.0, now - self._last)
        self._last = now
        return maintain(self.runtime, elapsed_seconds=elapsed)
