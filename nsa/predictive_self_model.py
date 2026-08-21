"""Learned predictor for explicit NSA self-state trajectories.

This module is a compatibility adapter for the Phase 19 trajectory experiment.
It deliberately predicts only the explicit seven-dimensional ``SelfState``
observation vector. It has no access to transformer hidden activations and no
authority over canonical hard state.
"""
from __future__ import annotations

import torch
from torch import nn

from nsa.self_model.core import ConditionedPredictiveSelfModel

SELF_STATE_FIELDS = (
    "confidence",
    "uncertainty",
    "perceived_risk",
    "capability_awareness",
    "resource_pressure",
    "goal_progress",
    "state_prediction_error",
)


class PredictiveSelfModel(nn.Module):
    """Predict the next explicit self-state from state and action features."""

    def __init__(self, state_dim: int = 7, action_dim: int = 4, hidden_dim: int = 32) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = ConditionedPredictiveSelfModel(
            d_model=1,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden_dim,
            immutable_index=None,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dimension {self.state_dim}, got {state.shape[-1]}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}, got {action.shape[-1]}")
        meaning = torch.zeros(
            state.shape[:-1] + (1,), device=state.device, dtype=state.dtype
        )
        return self.model(meaning, state, action)["predicted_state"]

    def training_loss(
        self, state: torch.Tensor, target: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        prediction = self(state, action)
        return torch.mean((prediction - target) ** 2)


__all__ = ["PredictiveSelfModel", "SELF_STATE_FIELDS"]
