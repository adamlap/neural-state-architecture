"""
nsa.self_model
==============
Predictive Self-Model, Internal Simulation, and Metacognition Primitives (Phases 18 & 19).

Features:
1. Conditioned Predictive Self-Model: P_theta(m_t, sigma_t, a_t) -> (sigma_hat_{t+1}, epsilon_hat_{t+1}, r_hat_{t+1})
   - Predicts transition delta conditioned on current state and candidate action.
   - Outputs calibrated uncertainty and predicted state improvement quality.
   - Enforces hard security coordinate immutability (sigma_h).
2. Counterfactual Internal Simulator:
   - Simulates candidate actions against internal state model.
   - Selects optimal legal transition maximizing recovery quality penalized by uncertainty:
       a* = argmax_{a in A_legal} [ r_hat(a) - lambda * epsilon_hat(a) ]
3. Backward-compatible primitives for existing cognitive and verifier pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RegulationDecision:
    """Bounded cognitive recommendation; never an authority mechanism."""

    confidence: torch.Tensor
    caution: torch.Tensor
    request_reassessment: torch.Tensor


@dataclass(frozen=True)
class SimulationResult:
    """Result of counterfactual internal simulation for a candidate action."""

    action_id: str
    predicted_state: torch.Tensor
    uncertainty: torch.Tensor
    predicted_quality: torch.Tensor
    is_legal: bool
    score: float


class ConditionedPredictiveSelfModel(nn.Module):
    """Predictive self-model conditioned on current state, meaning, and action context.
    
    Predicts:
    1. Future state transition: Delta sigma_hat
    2. Prediction uncertainty: epsilon_hat in [0, 1]
    3. Predicted recovery quality: r_hat in [-1, 1]
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        action_dim: Optional[int] = None,
        hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.action_dim = action_dim or state_dim
        hidden = hidden or max(d_model, state_dim * 4)

        in_dim = d_model + state_dim + self.action_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # 1. State transition delta head
        self.delta_head = nn.Linear(hidden, state_dim)

        # 2. Uncertainty head (predicts normalized error magnitude [0, 1])
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # 3. Recovery quality head (predicts state improvement [-1, 1])
        self.quality_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

        # Error projection for readout feedback
        self.error_projection = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
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

        inp = torch.cat((meaning, current_state, action_context), dim=-1)
        feat = self.trunk(inp)

        raw_delta = self.delta_head(feat)
        # Hard security coordinate is immutable: delta at index 0 is strictly zero
        delta = torch.cat((torch.zeros_like(raw_delta[..., :1]), raw_delta[..., 1:]), dim=-1)

        predicted_state = current_state + delta
        uncertainty = self.uncertainty_head(feat)
        predicted_quality = self.quality_head(feat)

        return {
            "predicted_delta": delta,
            "predicted_state": predicted_state,
            "uncertainty": uncertainty,
            "predicted_quality": predicted_quality,
        }

    def predict(
        self,
        meaning: torch.Tensor,
        current_state: torch.Tensor,
        action_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method returning predicted state."""
        out = self.forward(meaning, current_state, action_context)
        return out["predicted_state"]

    def prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        return actual - predicted


class CounterfactualInternalSimulator:
    """Internal simulator evaluating candidate transitions under uncertainty."""

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
        candidates: List[Tuple[str, torch.Tensor, bool]],  # (action_id, action_tensor, is_legally_permitted)
    ) -> Tuple[Optional[SimulationResult], List[SimulationResult]]:
        """Simulate each candidate action and select the optimal legal transition."""
        results: List[SimulationResult] = []
        best_result: Optional[SimulationResult] = None
        best_score = float("-inf")

        for action_id, action_tensor, is_legal in candidates:
            if not is_legal:
                res = SimulationResult(
                    action_id=action_id,
                    predicted_state=current_state,
                    uncertainty=torch.tensor([1.0]),
                    predicted_quality=torch.tensor([-1.0]),
                    is_legal=False,
                    score=float("-inf"),
                )
                results.append(res)
                continue

            with torch.no_grad():
                out = self.self_model(meaning, current_state, action_tensor)
                pred_state = out["predicted_state"]
                unc = out["uncertainty"].mean().item()
                qual = out["predicted_quality"].mean().item()

                score = qual - self.uncertainty_penalty * unc

                res = SimulationResult(
                    action_id=action_id,
                    predicted_state=pred_state,
                    uncertainty=out["uncertainty"],
                    predicted_quality=out["predicted_quality"],
                    is_legal=True,
                    score=score,
                )
                results.append(res)

                if score > best_score:
                    best_score = score
                    best_result = res

        return best_result, results


class PredictiveSelfState(nn.Module):
    """Backward-compatible predictive self-state module."""

    def __init__(self, d_model: int, state_dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or max(d_model, state_dim * 4)
        self.conditioned_model = ConditionedPredictiveSelfModel(
            d_model=d_model, state_dim=state_dim, action_dim=state_dim, hidden=hidden
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
        error = self.prediction_error(predicted, actual_next)
        return {
            "predicted_state": predicted,
            "prediction_error": error,
            "error_signal": self.error_projection(error),
            "prediction_mse": (error * error).mean(dim=-1, keepdim=True),
            "uncertainty": out["uncertainty"],
            "predicted_quality": out["predicted_quality"],
        }


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


__all__ = [
    "ConditionedPredictiveSelfModel",
    "CounterfactualInternalSimulator",
    "SimulationResult",
    "PredictiveSelfState",
    "SelfRegulationController",
    "CapabilityMonitor",
    "RegulationDecision",
]
