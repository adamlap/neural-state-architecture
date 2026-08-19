"""Predictive self-state and bounded self-regulation primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class PredictiveSelfState(nn.Module):
    """Predict the next structured state and expose prediction error."""

    def __init__(self, d_model: int, state_dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or max(d_model, state_dim * 4)
        self.predictor = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden), nn.GELU(), nn.Linear(hidden, state_dim)
        )
        self.error_projection = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, state_dim)
        )

    def predict(self, meaning: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat((meaning, state), dim=-1))

    def prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return actual - predicted

    def forward(self, meaning: torch.Tensor, state: torch.Tensor, actual_next: Optional[torch.Tensor] = None):
        predicted = self.predict(meaning, state)
        if actual_next is None:
            return {"predicted_state": predicted}
        error = self.prediction_error(predicted, actual_next)
        return {
            "predicted_state": predicted,
            "prediction_error": error,
            "error_signal": self.error_projection(error),
            "prediction_mse": (error * error).mean(dim=-1, keepdim=True),
        }


@dataclass(frozen=True)
class RegulationDecision:
    """Bounded cognitive recommendation; never an authorization decision."""

    confidence: torch.Tensor
    caution: torch.Tensor
    request_reassessment: torch.Tensor


class SelfRegulationController(nn.Module):
    """Turn self-model error into bounded cognitive caution signals."""

    def forward(self, prediction_error: torch.Tensor) -> RegulationDecision:
        uncertainty = prediction_error.pow(2).mean(dim=-1, keepdim=True)
        caution = torch.sigmoid(uncertainty)
        return RegulationDecision(
            confidence=1.0 - caution,
            caution=caution,
            request_reassessment=caution >= 0.7,
        )


class CapabilityMonitor(nn.Module):
    """Estimate capability; advisory only and never an authority mechanism."""

    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        hidden = max(16, d_model // 2)
        self.head = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, meaning: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((meaning, state), dim=-1))


__all__ = ["PredictiveSelfState", "SelfRegulationController", "CapabilityMonitor", "RegulationDecision"]
