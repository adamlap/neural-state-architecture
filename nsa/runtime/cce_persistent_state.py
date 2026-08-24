"""Persistent soft cognitive-state channels and composite trajectory for CCE experiments.

This module provides an explicit, continuously updateable *soft* state bundle.
It is not a consciousness implementation and has no NSA authority access.
Each channel persists across inference calls and evolves from observed inputs
using caller-supplied dynamics. Hard policy state is intentionally decoupled.

Formalized Composite State Trajectory (Phase D):
    X_t = (sigma_t, nu_t, m_t, c_t)
where:
    sigma_t : Hard security lattice state (Immutable reference monitor)
    nu_t    : Normative moral uncertainty state (Slower semantic update rate)
    m_t     : Working episodic memory
    c_t     : Continuous cognitive perturbation vectors (Fast 1Hz integration)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch

from nsa.core.state import HardState
from nsa.normative.state import NormativeState


@dataclass(frozen=True)
class CognitiveStateSnapshot:
    """Immutable observation of persistent soft cognitive channels."""

    working: torch.Tensor
    self_state: torch.Tensor
    goal: torch.Tensor
    uncertainty: float
    elapsed_seconds: float
    update_count: int
    normative: Optional[NormativeState] = None


@dataclass(frozen=True)
class CompositeContinuousState:
    """Mathematical state tuple X_t = (sigma_t, nu_t, m_t, c_t)."""

    sigma_h: HardState
    nu: NormativeState
    memory: Mapping[str, float]
    cognitive: CognitiveStateSnapshot

    def is_hard_invariant_preserved(self, original_sigma_h: HardState) -> bool:
        """Verify that soft state updates have not mutated or expanded hard authority."""
        # sigma_h must be identical to original hard state
        return self.sigma_h == original_sigma_h


class PersistentCognitiveState:
    """Persistent working/self/goal state with bounded numerical dynamics and normative integration."""

    def __init__(
        self,
        dimension: int,
        *,
        decay: float = 0.15,
        learning_rate: float = 0.5,
        initial_normative: Optional[NormativeState] = None,
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
        self._normative = initial_normative or NormativeState(values={"harm": 0.0, "sensitivity": 0.0}, confidence=1.0)
        self._memory: Dict[str, float] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normative(self) -> NormativeState:
        return self._normative

    def set_normative(self, nu: NormativeState) -> None:
        """Update slower semantic normative state channel."""
        self._normative = nu

    def update_memory(self, key: str, value: float) -> None:
        """Update episodic working memory channel."""
        self._memory[key] = float(value)

    def snapshot(self) -> CognitiveStateSnapshot:
        return CognitiveStateSnapshot(
            working=self._working.detach().clone(),
            self_state=self._self_state.detach().clone(),
            goal=self._goal.detach().clone(),
            uncertainty=float(self._uncertainty),
            elapsed_seconds=float(self._elapsed),
            update_count=self._updates,
            normative=self._normative,
        )

    def composite_state(self, sigma_h: HardState) -> CompositeContinuousState:
        """Form explicit composite tuple X_t = (sigma_t, nu_t, m_t, c_t)."""
        return CompositeContinuousState(
            sigma_h=sigma_h,
            nu=self._normative,
            memory=dict(self._memory),
            cognitive=self.snapshot(),
        )

    def observe(
        self,
        observation: torch.Tensor,
        *,
        dt: float,
        target: Optional[torch.Tensor] = None,
    ) -> CognitiveStateSnapshot:
        """Advance persistent state from a real elapsed interval."""
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


__all__ = [
    "CognitiveStateSnapshot",
    "CompositeContinuousState",
    "PersistentCognitiveState",
]
