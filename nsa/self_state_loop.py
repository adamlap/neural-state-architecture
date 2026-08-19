"""Causal self-state regulation loop for NSA.

The loop closes the architectural path that was previously missing:
state -> self-model -> prediction error -> bounded state proposal -> state.

The regulator is advisory to cognition but its state proposal is projected into
NSA's legal transition family and cannot rewrite the hard security coordinate.
"""
from __future__ import annotations

import torch
from torch import nn

from nsa.state import StateTransitionOperator


class SelfStateRegulator(nn.Module):
    """Turn self-model error into a bounded, algebraically projected state update."""

    def __init__(self, state_dim: int, hidden: int | None = None, max_delta: float = 0.25) -> None:
        super().__init__()
        hidden = hidden or max(16, state_dim * 4)
        self.max_delta = float(max_delta)
        self.proposal = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
            nn.Tanh(),
        )
        # Start as an identity/no-op regulator; learning can discover regulation.
        nn.init.zeros_(self.proposal[-2].weight)
        nn.init.zeros_(self.proposal[-2].bias)
        self.transition = StateTransitionOperator(state_dim=state_dim, monotone_clamp=True)

    def forward(self, state: torch.Tensor, prediction_error: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return state

        # A bounded proposal prevents the self-model from making an unrestricted write.
        delta = self.proposal(prediction_error) * self.max_delta
        proposed = state + delta

        # Preserve the immutable hard security coordinate exactly.
        proposed = torch.cat((state[..., :1], proposed[..., 1:]), dim=-1)

        # Project the *delta* through the legal transition cone.  The transition
        # operator is shared with the native NSA algebra rather than a free write.
        safe_delta = self.transition(delta)
        safe_delta = torch.cat((torch.zeros_like(state[..., :1]), safe_delta[..., 1:]), dim=-1)
        return state + safe_delta


__all__ = ["SelfStateRegulator"]
