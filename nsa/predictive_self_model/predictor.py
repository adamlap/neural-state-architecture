"""Learned predictive model over explicit NSA computational self-state.

This module predicts the *explicit* ``SelfState`` representation. It does not
access or claim to predict transformer hidden activations. Predictions are
observational and cannot mutate trusted runtime or hard authority state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from nsa.self_state.model import SelfState


SELF_STATE_FIELDS = (
    "confidence",
    "uncertainty",
    "perceived_risk",
    "capability_awareness",
    "resource_pressure",
    "goal_progress",
    "state_prediction_error",
)


@dataclass(frozen=True)
class PredictionResult:
    """Prediction plus supervised error against an observed future state."""

    predicted: SelfState
    mse: float | None = None

    def compare(self, observed: SelfState) -> "PredictionResult":
        target = torch.tensor([_state_values(observed)], dtype=torch.float32)
        pred = torch.tensor([_state_values(self.predicted)], dtype=torch.float32)
        return PredictionResult(self.predicted, float(torch.mean((pred - target) ** 2)))


class PredictiveSelfModel(nn.Module):
    """Small trainable dynamics model for explicit NSA self-state.

    ``state_dim`` is accepted explicitly so trajectory experiments can bind the
    model to the schema they are evaluating. NSA currently defines exactly
    seven explicit self-state dimensions; any other value is rejected rather
    than silently reshaping the state space.
    """

    state_dim = len(SELF_STATE_FIELDS)

    def __init__(
        self,
        state_dim: int | None = None,
        action_dim: int = 0,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        resolved_state_dim = self.state_dim if state_dim is None else int(state_dim)
        if resolved_state_dim != self.state_dim:
            raise ValueError(
                f"NSA PredictiveSelfModel requires state_dim={self.state_dim}, "
                f"got {resolved_state_dim}"
            )
        if action_dim < 0:
            raise ValueError("action_dim must be non-negative")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.state_dim),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        state = _validate_batch(state, self.state_dim, "state")
        if self.action_dim:
            if action is None:
                raise ValueError("action is required when action_dim > 0")
            action = _validate_batch(action, self.action_dim, "action")
            if action.shape[0] != state.shape[0]:
                raise ValueError("state and action batch sizes must match")
            features = torch.cat((state, action), dim=-1)
        else:
            if action is not None and action.numel():
                raise ValueError("action supplied to a zero-action model")
            features = state
        return self.net(features)

    @torch.no_grad()
    def predict(self, state: SelfState, action: Iterable[float] | None = None) -> PredictionResult:
        state_tensor = torch.tensor([_state_values(state)], dtype=torch.float32)
        action_tensor = None
        if self.action_dim:
            if action is None:
                raise ValueError("action is required when action_dim > 0")
            values = list(action)
            if len(values) != self.action_dim:
                raise ValueError("action length does not match action_dim")
            action_tensor = torch.tensor([values], dtype=torch.float32)
        predicted = self.forward(state_tensor, action_tensor)[0].tolist()
        return PredictionResult(_state_from_values(predicted, step=state.step + 1))

    def training_loss(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = _validate_batch(target, self.state_dim, "target")
        prediction = self.forward(state, action)
        return torch.mean((prediction - target) ** 2)


def _state_values(state: SelfState) -> list[float]:
    return [float(getattr(state, field)) for field in SELF_STATE_FIELDS]


def _state_from_values(values: list[float], *, step: int) -> SelfState:
    bounded = [min(1.0, max(0.0, float(v))) for v in values]
    return SelfState(**dict(zip(SELF_STATE_FIELDS, bounded)), step=step)


def _validate_batch(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch, {width}]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value.float()


__all__ = ["PredictiveSelfModel", "PredictionResult", "SELF_STATE_FIELDS"]
