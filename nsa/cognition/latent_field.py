"""Continuous latent cognitive field and sub-inference thought vectors.

Enables continuous, high-frequency (10Hz-50Hz) internal cognitive dynamics
without requiring full autoregressive token generation on every wall-clock tick.
Maintains bounded Lipschitz dynamics and projects continuous latent state
into NSA SoftState features and deliberation triggers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import monotonic
from typing import Optional, Tuple
import torch

from nsa.core.state import SoftState


@dataclass(frozen=True)
class LatentThoughtVector:
    """Continuous internal thought representation at a single instant."""
    vector: torch.Tensor
    step: int
    timestamp: float = field(default_factory=monotonic)

    @property
    def dimension(self) -> int:
        return self.vector.shape[-1]

    @property
    def energy(self) -> float:
        return float(torch.linalg.norm(self.vector).item())

    def to_tuple(self) -> Tuple[float, ...]:
        return tuple(round(x, 4) for x in self.vector.tolist())


@dataclass(frozen=True)
class LatentFieldConfig:
    dimension: int = 128
    integration_rate: float = 0.20  # alpha in leaky integrator
    decay_rate: float = 0.02
    max_norm: float = 10.0  # Lipschitz stability bound
    deliberation_threshold: float = 0.65  # triggers explicit deliberation/action
    seed: int = 42


class LatentCognitiveField:
    """Continuous dynamical cognitive field evolving at sub-inference timescales."""

    def __init__(self, config: Optional[LatentFieldConfig] = None) -> None:
        self.config = config or LatentFieldConfig()
        torch.manual_seed(self.config.seed)
        # Initialize internal projection matrices
        self.d = self.config.dimension
        # Stable internal recurrent transition matrix (orthogonal init for stable dynamics)
        q, _ = torch.linalg.qr(torch.randn(self.d, self.d))
        self._W_rec = q * 0.95  # Spectral radius < 1 guarantees stability
        # Soft state projection heads (fixed linear readouts)
        self._proj_uncertainty = torch.randn(self.d) / math.sqrt(self.d)
        self._proj_risk = torch.randn(self.d) / math.sqrt(self.d)
        self._proj_resource = torch.randn(self.d) / math.sqrt(self.d)

        # Initial state
        self._current_vec = torch.zeros(self.d, dtype=torch.float32)
        self._step = 0
        self._history: list[LatentThoughtVector] = []

    @property
    def current(self) -> LatentThoughtVector:
        return LatentThoughtVector(
            vector=self._current_vec.clone(),
            step=self._step,
            timestamp=monotonic(),
        )

    @staticmethod
    def _fit_vector(value: torch.Tensor, dimension: int) -> torch.Tensor:
        value = value.detach().to(dtype=torch.float32).reshape(-1)
        if value.numel() == dimension:
            return value
        if value.numel() == 0:
            return torch.zeros(dimension, dtype=torch.float32)
        return torch.nn.functional.interpolate(
            value.view(1, 1, -1), size=dimension, mode="linear", align_corners=False
        ).reshape(-1)

    def tick(
        self,
        sensory_input: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> LatentThoughtVector:
        """Advance the continuous thought vector by one sub-inference time step.

        Equation: z_{t+1} = (1 - alpha - decay) * z_t + alpha * (W_rec @ z_t + input) + drift
        """
        self._step += 1
        alpha = self.config.integration_rate
        decay = self.config.decay_rate

        # Recurrent association
        rec = torch.matmul(self._W_rec, self._current_vec)

        # Sensory integration
        input_term = self._fit_vector(sensory_input, self.d) if sensory_input is not None else torch.zeros_like(self._current_vec)
        drift_term = self._fit_vector(drift, self.d) if drift is not None else torch.zeros_like(self._current_vec)

        # Dynamical integration step
        updated = (1.0 - alpha - decay) * self._current_vec + alpha * (rec + input_term) + drift_term

        # Lipschitz energy bounding
        norm = torch.linalg.norm(updated)
        if norm > self.config.max_norm:
            updated = (updated / norm) * self.config.max_norm

        self._current_vec = updated.detach()
        state = self.current
        self._history.append(state)
        if len(self._history) > 1000:
            self._history.pop(0)
        return state

    def project_soft_state(self) -> SoftState:
        """Project continuous latent thought vector into NSA canonical SoftState."""
        z = self._current_vec
        # Map through sigmoid to [0, 1] range
        raw_u = float(torch.sigmoid(torch.dot(z, self._proj_uncertainty)).item())
        raw_r = float(torch.sigmoid(torch.dot(z, self._proj_risk)).item())
        raw_p = float(torch.sigmoid(torch.dot(z, self._proj_resource)).item())
        confidence = max(0.0, min(1.0, 1.0 - raw_u))

        return SoftState(
            uncertainty=max(0.0, min(1.0, raw_u)),
            risk=max(0.0, min(1.0, raw_r)),
            confidence=max(0.0, min(1.0, confidence)),
            resource_pressure=max(0.0, min(1.0, raw_p)),
        )

    def requires_deliberation(self) -> bool:
        """True when internal cognitive dissonance or uncertainty warrants an explicit deliberation step."""
        soft = self.project_soft_state()
        return soft.uncertainty >= self.config.deliberation_threshold or soft.risk >= self.config.deliberation_threshold

    def inject_thought_perturbation(self, perturbation: torch.Tensor) -> None:
        """Inject a targeted thought or goal shift into the latent field."""
        perturbation = self._fit_vector(perturbation, self.d)
        self._current_vec = self._current_vec + perturbation
        norm = torch.linalg.norm(self._current_vec)
        if norm > self.config.max_norm:
            self._current_vec = (self._current_vec / norm) * self.config.max_norm

    def reset(self) -> None:
        self._current_vec = torch.zeros(self.d, dtype=torch.float32)
        self._step = 0
        self._history.clear()