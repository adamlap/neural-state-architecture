"""Learned one-step predictors for CCE continuous state dynamics.

This module intentionally separates *learning* a state transition from the
continuous runtime that integrates it. A predictor estimates the next state
from the current state and optional input. It is never an authority boundary.

The reference implementation is a small PyTorch MLP suitable for deterministic
CI experiments. It can be trained on recorded trajectories and evaluated
against a persistence baseline before being admitted to the CCE field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class PredictionMetrics:
    mse: float
    persistence_mse: float
    improvement: float


class StatePredictor(nn.Module):
    """Predict the next state from the current state and optional input."""

    def __init__(self, state_dim: int, input_dim: int = 0, hidden_dim: int = 64) -> None:
        super().__init__()
        if state_dim <= 0 or input_dim < 0 or hidden_dim <= 0:
            raise ValueError("state_dim > 0, input_dim >= 0 and hidden_dim > 0 required")
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, external: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError("state has incompatible final dimension")
        if self.input_dim:
            if external is None or external.shape[-1] != self.input_dim:
                raise ValueError("external input is required with the configured input_dim")
            x = torch.cat((state, external), dim=-1)
        else:
            x = state
        return self.net(x)


def prediction_metrics(predicted: torch.Tensor, target: torch.Tensor, current: torch.Tensor) -> PredictionMetrics:
    """Compare a learned predictor against the persistence baseline."""
    if predicted.shape != target.shape or current.shape != target.shape:
        raise ValueError("predicted, target and current must have identical shapes")
    mse = float(torch.mean((predicted - target) ** 2).detach().cpu())
    persistence = float(torch.mean((current - target) ** 2).detach().cpu())
    improvement = 0.0 if persistence == 0.0 else (persistence - mse) / persistence
    return PredictionMetrics(mse=mse, persistence_mse=persistence, improvement=improvement)


def train_predictor(
    model: StatePredictor,
    states: torch.Tensor,
    targets: torch.Tensor,
    external: Optional[torch.Tensor] = None,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> PredictionMetrics:
    """Train on a recorded trajectory and return training-set metrics.

    This is intentionally an explicit opt-in training operation; inference in
    CCE never mutates model weights. Callers should use a held-out trajectory
    for scientific evaluation before deploying a predictor.
    """
    if states.ndim != 2 or targets.shape != states.shape:
        raise ValueError("states and targets must be [samples, state_dim]")
    if states.shape[0] < 2:
        raise ValueError("at least two trajectory samples are required")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if external is not None and external.shape[0] != states.shape[0]:
        raise ValueError("external must have the same number of samples as states")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = model(states, external)
        loss = torch.mean((predicted - targets) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted = model(states, external)
    return prediction_metrics(predicted, targets, states)
