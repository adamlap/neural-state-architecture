"""Bridge a validated next-state predictor into the continuous CCE field.

The predictor is defined at a declared reference interval.  This adapter
converts its next-state estimate into a derivative so the existing
ContinuousStateField remains responsible for wall-clock integration,
threading, finite-value checks, and fail-closed behavior.

The adapter has no authority access and is never enabled implicitly.
"""
from __future__ import annotations

from typing import Optional

import torch

from .predictive_dynamics import StatePredictor


class PredictiveDynamicsField:
    """Callable continuous field backed by a learned next-state predictor."""

    def __init__(self, predictor: StatePredictor, *, reference_dt: float = 0.01, enabled: bool = False) -> None:
        if reference_dt <= 0:
            raise ValueError("reference_dt must be > 0")
        self.predictor = predictor
        self.reference_dt = float(reference_dt)
        self.enabled = bool(enabled)

    def __call__(self, state: torch.Tensor, external: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(state)
        with torch.no_grad():
            predicted = self.predictor(state, external)
        if predicted.shape != state.shape:
            raise ValueError("predictor returned an incompatible state shape")
        if not torch.isfinite(predicted).all():
            raise ValueError("predictor returned non-finite state")
        return (predicted - state) / self.reference_dt

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
