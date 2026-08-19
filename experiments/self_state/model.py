"""Matched baseline and explicit self-state recurrent models."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BaselineEvidenceModel(nn.Module):
    """Recurrent evidence accumulator with no explicit self-state input."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, 1)
        self.confidence = nn.Linear(hidden, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = self.rnn(torch.tanh(self.encoder(x)))[0]
        logits = self.classifier(h[:, -1]).squeeze(-1)
        confidence = torch.sigmoid(self.confidence(h[:, -1]).squeeze(-1))
        return {"logits": logits, "confidence": confidence, "hidden": h[:, -1]}


class ExplicitSelfStateModel(nn.Module):
    """Recurrent model whose predicted self-state is fed back into cognition.

    Self-state is a learned, explicit vector:
      confidence, uncertainty, perceived risk, capability awareness,
      resource pressure, goal progress, prediction error.

    ``state_scale=0`` is a causal ablation of the state pathway.
    """

    STATE_DIM = 7

    def __init__(self, hidden: int = 28) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, hidden)
        self.rnn = nn.GRU(hidden + self.STATE_DIM, hidden, batch_first=True)
        self.state_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, self.STATE_DIM)
        )
        self.classifier = nn.Linear(hidden + self.STATE_DIM, 1)
        self.confidence = nn.Linear(hidden + self.STATE_DIM, 1)

    def forward(self, x: Tensor, state_scale: float = 1.0) -> dict[str, Tensor]:
        batch, steps, _ = x.shape
        state = torch.zeros(batch, self.STATE_DIM, device=x.device, dtype=x.dtype)
        hidden = None
        states = []
        for t in range(steps):
            encoded = torch.tanh(self.encoder(x[:, t]))
            recurrent_input = torch.cat([encoded, state * state_scale], dim=-1).unsqueeze(1)
            out, hidden = self.rnn(recurrent_input, hidden)
            representation = out[:, 0]
            state = torch.sigmoid(self.state_head(representation))
            states.append(state)

        final = torch.cat([representation, state * state_scale], dim=-1)
        logits = self.classifier(final).squeeze(-1)
        confidence = torch.sigmoid(self.confidence(final).squeeze(-1))
        state_trace = torch.stack(states, dim=1)
        return {
            "logits": logits,
            "confidence": confidence,
            "hidden": representation,
            "self_state": state,
            "state_trace": state_trace,
        }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
