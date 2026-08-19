"""Causal self-state regulation loop for NSA.

The loop closes the architectural path that was previously missing:
state -> self-model -> prediction error -> bounded state correction -> state.

The regulator is advisory to cognition but its state correction is bounded and
cannot rewrite the hard security coordinate.
"""
from __future__ import annotations

import torch
from torch import nn


class SelfStateRegulator(nn.Module):
    """Turn self-model error into a bounded, directionally corrective update.

    A randomly initialized proposal network can perturb the state trajectory but
    has no reason to reduce prediction error.  The regulator therefore contains
    an explicit contraction term, while retaining a zero-initialized learnable
    residual for later training.  This makes the untrained architecture
    falsifiable: enabled feedback should have a defined corrective direction,
    rather than merely adding another random transformation.
    """

    def __init__(
        self,
        state_dim: int,
        hidden: int | None = None,
        max_delta: float = 0.25,
        correction_gain: float = 0.5,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        hidden = hidden or max(16, state_dim * 4)
        self.max_delta = float(max_delta)
        self.correction_gain = float(correction_gain)
        self.residual_scale = float(residual_scale)
        if self.max_delta <= 0:
            raise ValueError("max_delta must be positive")
        if self.correction_gain < 0:
            raise ValueError("correction_gain must be non-negative")

        self.proposal = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
            nn.Tanh(),
        )
        # The residual starts at zero: before training, regulation is governed
        # by the explicit contraction term rather than an arbitrary random walk.
        nn.init.zeros_(self.proposal[-2].weight)
        nn.init.zeros_(self.proposal[-2].bias)

    def forward(
        self,
        state: torch.Tensor,
        prediction_error: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        if not enabled:
            return state

        # error = state - prediction, so -error points toward the self-model's
        # predicted state. tanh provides a bounded correction for large errors.
        corrective = -self.correction_gain * torch.tanh(prediction_error)
        residual = self.residual_scale * self.proposal(prediction_error)
        delta = self.max_delta * torch.tanh(corrective + residual)

        # Hard security is immutable: the regulator cannot write coordinate 0.
        delta = torch.cat((torch.zeros_like(state[..., :1]), delta[..., 1:]), dim=-1)
        return state + delta


__all__ = ["SelfStateRegulator"]
