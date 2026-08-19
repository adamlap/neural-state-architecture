"""Causal self-state regulation loop for NSA.

The loop closes the architectural path that was previously missing:
state -> self-model -> prediction error -> bounded state proposal -> state.

The regulator is advisory to cognition but its state proposal is bounded and
cannot rewrite the hard security coordinate.
"""
from __future__ import annotations

import torch
from torch import nn


class SelfStateRegulator(nn.Module):
    """Turn self-model error into a bounded state update."""

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
        # Small non-zero initialization makes the closed loop observable before
        # training while keeping the architectural intervention deliberately weak.
        nn.init.normal_(self.proposal[-2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.proposal[-2].bias)

    def forward(self, state: torch.Tensor, prediction_error: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return state
        delta = self.proposal(prediction_error) * self.max_delta
        # Hard security is immutable: the regulator cannot write coordinate 0.
        delta = torch.cat((torch.zeros_like(state[..., :1]), delta[..., 1:]), dim=-1)
        return state + delta


__all__ = ["SelfStateRegulator"]
