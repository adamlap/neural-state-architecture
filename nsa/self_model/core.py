"""Predictive self-model primitives (package implementation)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class RegulationDecision:
    confidence: torch.Tensor
    caution: torch.Tensor
    request_reassessment: torch.Tensor


@dataclass(frozen=True)
class SimulationResult:
    action_id: str
    predicted_state: torch.Tensor
    uncertainty: torch.Tensor
    predicted_quality: torch.Tensor
    is_legal: bool
    score: float


class ConditionedPredictiveSelfModel(nn.Module):
    """Predict state transitions with an explicit immutable-coordinate policy."""

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden: Optional[int] = None,
        immutable_index: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.action_dim = action_dim or state_dim
        self.immutable_index = immutable_index
        if immutable_index is not None and not 0 <= immutable_index < state_dim:
            raise ValueError("immutable_index must be None or a valid state coordinate")
        hidden = hidden or max(d_model, state_dim * 4)
        self.trunk = nn.Sequential(
            nn.Linear(d_model + state_dim + self.action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden, state_dim)
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.quality_head = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())
        self.error_projection = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, state_dim)
        )

    def forward(
        self,
        meaning: torch.Tensor,
        current_state: torch.Tensor,
        action_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if action_context is None:
            action_context = torch.zeros(
                meaning.shape[:-1] + (self.action_dim,),
                device=meaning.device,
                dtype=meaning.dtype,
            )
        feat = self.trunk(torch.cat((meaning, current_state, action_context), dim=-1))
        raw_delta = self.delta_head(feat)
        if self.immutable_index is None:
            delta = raw_delta
        else:
            delta = raw_delta.clone()
            delta[..., self.immutable_index] = 0.0
        return {
            "predicted_delta": delta,
            "predicted_state": current_state + delta,
            "uncertainty": self.uncertainty_head(feat),
            "predicted_quality": self.quality_head(feat),
        }

    def predict(
        self,
        meaning: torch.Tensor,
        current_state: torch.Tensor,
        action_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(meaning, current_state, action_context)["predicted_state"]

    def prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return actual - predicted


class CounterfactualInternalSimulator:
    def __init__(
        self,
        self_model: ConditionedPredictiveSelfModel,
        uncertainty_penalty: float = 0.5,
    ) -> None:
        self.self_model = self_model
        self.uncertainty_penalty = float(uncertainty_penalty)

    def evaluate_candidates(
        self,
        meaning: torch.Tensor,
        current_state: torch.Tensor,
        candidates: List[Tuple[str, torch.Tensor, bool]],
    ) -> Tuple[Optional[SimulationResult], List[SimulationResult]]:
        results: List[SimulationResult] = []
        best: Optional[SimulationResult] = None
        best_score = float("-inf")
        for action_id, action_tensor, is_legal in candidates:
            if not is_legal:
                results.append(
                    SimulationResult(
                        action_id,
                        current_state,
                        torch.tensor([1.0]),
                        torch.tensor([-1.0]),
                        False,
                        float("-inf"),
                    )
                )
                continue
            with torch.no_grad():
                out = self.self_model(meaning, current_state, action_tensor)
                unc = out["uncertainty"].mean().item()
                qual = out["predicted_quality"].mean().item()
                score = qual - self.uncertainty_penalty * unc
                result = SimulationResult(
                    action_id,
                    out["predicted_state"],
                    out["uncertainty"],
                    out["predicted_quality"],
                    True,
                    score,
                )
                results.append(result)
                if score > best_score:
                    best_score = score
                    best = result
        return best, results


class PredictiveSelfState(nn.Module):
    def __init__(self, d_model: int, state_dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or max(d_model, state_dim * 4)
        self.conditioned_model = ConditionedPredictiveSelfModel(
            d_model, state_dim, state_dim, hidden, immutable_index=0
        )
        self.predictor = self.conditioned_model
        self.error_projection = self.conditioned_model.error_projection

    def predict(self, meaning: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.conditioned_model.predict(meaning, state)

    def prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return actual - predicted

    def forward(
        self,
        meaning: torch.Tensor,
        state: torch.Tensor,
        actual_next: Optional[torch.Tensor] = None,
    ):
        out = self.conditioned_model(meaning, state)
        predicted = out["predicted_state"]
        if actual_next is None:
            return {"predicted_state": predicted, "uncertainty": out["uncertainty"]}
        error = actual_next - predicted
        return {
            "predicted_state": predicted,
            "prediction_error": error,
            "error_signal": self.error_projection(error),
            "prediction_mse": (error * error).mean(dim=-1, keepdim=True),
            "uncertainty": out["uncertainty"],
            "predicted_quality": out["predicted_quality"],
        }


class SelfRegulationController(nn.Module):
    def forward(self, prediction_error: torch.Tensor) -> RegulationDecision:
        uncertainty = prediction_error.pow(2).mean(dim=-1, keepdim=True)
        caution = torch.sigmoid(uncertainty)
        return RegulationDecision(1.0 - caution, caution, caution >= 0.7)


class CapabilityMonitor(nn.Module):
    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        hidden = max(16, d_model // 2)
        self.head = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, meaning: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((meaning, state), dim=-1))


__all__ = [
    "ConditionedPredictiveSelfModel",
    "CounterfactualInternalSimulator",
    "SimulationResult",
    "PredictiveSelfState",
    "SelfRegulationController",
    "CapabilityMonitor",
    "RegulationDecision",
]
