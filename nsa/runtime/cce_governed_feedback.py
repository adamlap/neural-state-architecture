"""Bounded feedback proposals for the CCE soft cognitive state.

LLM output is treated as untrusted data. This module validates a typed proposal,
clips its requested change to a configured budget, and applies it only through
the existing PersistentCognitiveState observation transition. It has no access
to NSA hard authority state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nsa.runtime.cce_persistent_state import CognitiveStateSnapshot, PersistentCognitiveState


@dataclass(frozen=True)
class CognitiveFeedbackProposal:
    """Untrusted, typed proposal for a bounded soft-state adjustment."""

    working_delta: tuple[float, ...]
    goal_delta: tuple[float, ...] = ()
    confidence: float = 0.0
    source: str = "external"


@dataclass(frozen=True)
class FeedbackResult:
    """Auditable result of applying a validated bounded proposal."""

    accepted: bool
    clipped_norm: float
    snapshot: CognitiveStateSnapshot


def _finite_vector(values: Sequence[float], dimension: int, name: str) -> torch.Tensor:
    if len(values) != dimension:
        raise ValueError(f"{name} dimension does not match state")
    tensor = torch.tensor(tuple(float(v) for v in values), dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite")
    return tensor


class GovernedCognitiveFeedback:
    """Apply only bounded proposals to the soft CCE state."""

    def __init__(self, state: PersistentCognitiveState, *, max_norm: float = 0.25) -> None:
        if max_norm <= 0.0:
            raise ValueError("max_norm must be > 0")
        self._state = state
        self._max_norm = float(max_norm)

    def apply(
        self,
        proposal: CognitiveFeedbackProposal,
        *,
        dt: float,
    ) -> FeedbackResult:
        if not 0.0 <= float(proposal.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(proposal.source, str) or not proposal.source.strip():
            raise ValueError("source must be non-empty")
        snapshot = self._state.snapshot()
        working_delta = _finite_vector(proposal.working_delta, self._state.dimension, "working_delta")
        goal_values = proposal.goal_delta or tuple(0.0 for _ in range(self._state.dimension))
        goal_delta = _finite_vector(goal_values, self._state.dimension, "goal_delta")
        # Confidence scales the untrusted request before the hard numerical budget.
        requested = torch.cat((working_delta, goal_delta)) * float(proposal.confidence)
        norm = float(torch.linalg.vector_norm(requested).item())
        scale = min(1.0, self._max_norm / norm) if norm > 0.0 else 1.0
        bounded = requested * scale
        half = self._state.dimension
        target_working = snapshot.working + bounded[:half]
        target_goal = snapshot.goal + bounded[half:]
        updated = self._state.observe(target_working, dt=dt, target=target_goal)
        return FeedbackResult(
            accepted=True,
            clipped_norm=float(torch.linalg.vector_norm(bounded).item()),
            snapshot=updated,
        )


__all__ = ["CognitiveFeedbackProposal", "FeedbackResult", "GovernedCognitiveFeedback"]
