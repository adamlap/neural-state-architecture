"""Synthetic sequential evidence task for the NSA self-state experiment."""

from __future__ import annotations

import torch
from torch import Tensor


def make_batch(
    batch_size: int,
    steps: int = 12,
    noise: float = 0.8,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate binary hypotheses observed through noisy sequential evidence.

    Each sample has a hidden class y in {-1,+1}. Every observation is a noisy
    measurement of y. The task therefore has an objectively changing evidence
    state: confidence should rise with consistent evidence and fall when the
    evidence is contradictory/noisy.
    """
    y = torch.randint(0, 2, (batch_size,), generator=generator, device=device).float()
    sign = y.mul(2.0).sub(1.0)
    observations = sign[:, None] + noise * torch.randn(
        batch_size, steps, generator=generator, device=device
    )
    # The model receives only observations; the true label is withheld.
    return observations.unsqueeze(-1), y


def make_shifted_batch(
    batch_size: int,
    steps: int = 12,
    noise: float = 1.8,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Harder distribution used only for out-of-distribution evaluation."""
    return make_batch(batch_size, steps, noise, device, generator)
