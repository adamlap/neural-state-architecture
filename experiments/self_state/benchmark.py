"""NSA Cognitive Benchmark v1.

Compares a baseline recurrent evidence accumulator, an NSA state-feedback
model, and the same NSA model with its state pathway causally ablated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BenchmarkCase:
    observations: torch.Tensor
    target: float
    difficulty: str


def make_cases(n: int = 256, noise: float = 0.75, seed: int = 0) -> list[BenchmarkCase]:
    generator = torch.Generator().manual_seed(seed)
    cases: list[BenchmarkCase] = []
    for _ in range(n):
        target = 1.0 if torch.rand((), generator=generator).item() >= 0.5 else -1.0
        evidence = target + noise * torch.randn(12, generator=generator)
        cases.append(BenchmarkCase(evidence, target, "iid"))
    return cases


class BaselineAccumulator(nn.Module):
    def __init__(self, hidden: int = 28) -> None:
        super().__init__()
        self.cell = nn.GRUCell(1, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(observations.shape[0], self.cell.hidden_size, device=observations.device)
        for step in observations.unbind(1):
            h = self.cell(step.unsqueeze(-1), h)
        return self.head(h).squeeze(-1)


class NSAAccumulator(nn.Module):
    """Matched recurrent model with an explicit seven-dimensional state path."""

    def __init__(self, hidden: int = 28, state_dim: int = 7) -> None:
        super().__init__()
        self.cell = nn.GRUCell(1, hidden)
        self.state = nn.Sequential(nn.Linear(hidden + 1, state_dim), nn.Tanh())
        self.feedback = nn.Linear(state_dim, hidden, bias=False)
        self.head = nn.Linear(hidden + state_dim, 1)
        self.state_dim = state_dim

    def forward(self, observations: torch.Tensor, state_scale: float = 1.0) -> torch.Tensor:
        h = torch.zeros(observations.shape[0], self.cell.hidden_size, device=observations.device)
        s = torch.zeros(observations.shape[0], self.state_dim, device=observations.device)
        for step in observations.unbind(1):
            h = self.cell(step.unsqueeze(-1), h)
            s = self.state(torch.cat((h, step.unsqueeze(-1)), dim=-1))
            h = h + state_scale * self.feedback(s)
        return self.head(torch.cat((h, s), dim=-1)).squeeze(-1)


def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits)
    labels = (targets > 0).float()
    return float(((probabilities - labels) ** 2).mean())


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = torch.where(logits >= 0, 1.0, -1.0)
    return float((predictions == targets).float().mean())


def evaluate(model: nn.Module, cases: Iterable[BenchmarkCase], *, state_scale: float = 1.0) -> dict[str, float]:
    batch = list(cases)
    observations = torch.stack([c.observations for c in batch]).float()
    targets = torch.tensor([c.target for c in batch]).float()
    with torch.no_grad():
        if isinstance(model, NSAAccumulator):
            logits = model(observations, state_scale=state_scale)
        else:
            logits = model(observations)
    return {"accuracy": accuracy(logits, targets), "brier": brier_score(logits, targets)}


__all__ = ["BenchmarkCase", "BaselineAccumulator", "NSAAccumulator", "make_cases", "evaluate"]
