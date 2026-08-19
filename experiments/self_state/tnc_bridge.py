"""Bridge the self-state benchmark to NSA's existing state-transition algebra.

This adapter uses the same projected transition semantics as the existing NSA
core rather than creating a parallel security mechanism.
"""
from __future__ import annotations

import torch
from torch import nn

from nsa.state import StateTransitionOperator


class TNCStateFeedback(nn.Module):
    """Project a learned self-state through NSA's legal transition cone."""

    def __init__(self, state_dim: int = 7) -> None:
        super().__init__()
        self.transition = StateTransitionOperator(state_dim=state_dim, monotone_clamp=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.transition(state)

    def projected_transition(self) -> torch.Tensor:
        return self.transition.get_projected_V()

    def illegal_mass(self) -> torch.Tensor:
        """Magnitude of raw parameters outside the legal lower triangle."""
        raw = self.transition.V
        return torch.triu(raw, diagonal=1).abs().sum()


__all__ = ["TNCStateFeedback"]
