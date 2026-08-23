"""Persistent soft cognitive-state channels for CCE experiments.

This module provides an explicit, continuously updateable *soft* state bundle.
It is not a consciousness implementation and has no NSA authority access.
Each channel persists across inference calls and evolves from observed inputs
using caller-supplied dynamics. Hard policy state is intentionally absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class CognitiveStateSnapshot:
    """Immutable observation of persistent soft cognitive channels."""

    working: torch.Tensor
    self_state: torch.Tensor
    goal: torch.Tensor
    uncertainty: float
    elapsed_seconds: float
    update_count: int


class PersistentCognitiveState:
    """Persistent working/self/goal state with bounded numerical dynamics.

    ``observe`` updates the channels using measured elapsed time. The update is
    intentionally generic: external observations are projected into the
    channel dimensions and blended with prior state. No language-model output
    is applied automatically and no hard NSA state is stored here.
    """

    def __init__(
        self,
        dimension: int,
        *,
        decay: float = 0.15,
        learning_rate: float = 0.5,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if learning_rate < 0.0:
            raise ValueError("learning_rate must be >= 0")
        self._dimension = int(dimension)
        self._decay = float(decay)
        self._learning_rate = float(learning_rate)
        self._working = torch.zeros(dimension)
        self._self_state = torch.zeros(dimension)
        self._goal = torch.zeros(dimension)
        self._uncertainty = 1.0
        self._elapsed = 0.0
        self._updates = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def snapshot(self) -> CognitiveStateSnapshot:
        return CognitiveStateSnapshot(
            working=self._working.detach().clone(),
            self_state=self._self_state.detach().clone(),
            goal=self._goal.detach().clone(),
            uncertainty=float(self._uncertainty),
            elapsed_seconds=float(self._elapsed),
            update_count=self._updates,
        )

    def observe(
        self,
        observation: torch.Tensor,
        *,
        dt: float,
        target: Optional[torch.Tensor] = None,
    ) -> CognitiveStateSnapshot:
        """Advance persistent state from a real elapsed interval.

        ``target`` is optional and represents a caller-owned goal signal. Both
        inputs are detached and shape-checked; non-finite values fail closed.
        """
        if dt < 0.0:
            raise ValueError("dt must be >= 0")
        obs = observation.detach().flatten().to(dtype=self._working.dtype)
        if obs.numel() != self._dimension:
            raise ValueError("observation dimension does not match state")
        if not torch.isfinite(obs).all():
            raise ValueError("observation must be finite")
        goal = self._goal if target is None else target.detach().flatten().to(dtype=self._goal.dtype)
        if goal.numel() != self._dimension:
            raise ValueError("target dimension does not match state")
        if not torch.isfinite(goal).all():
            raise ValueError("target must be finite")

        alpha = min(1.0, self._learning_rate * float(dt))
        decay = min(1.0, self._decay * float(dt))
        self._working = self._working + (obs - self._working) * alpha
        self._self_state = self._self_state * max(0.0, 1.0 - decay) + self._working * decay
        self._goal = self._goal + (goal - self._goal) * alpha
        mismatch = float(torch.linalg.vector_norm(obs - self._working).item())
        self._uncertainty = min(1.0, max(0.0, (1.0 - alpha) * self._uncertainty + alpha * min(1.0, mismatch)))
        self._elapsed += float(dt)
        self._updates += 1
        return self.snapshot()


__all__ = ["CognitiveStateSnapshot", "PersistentCognitiveState"]
