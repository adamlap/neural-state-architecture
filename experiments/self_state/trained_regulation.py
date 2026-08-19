"""Train the self-state predictor before evaluating bounded regulation.

This experiment freezes the native NSA model, fits the predictive self-state
module on native state trajectories, then evaluates the existing regulator on
controlled state perturbations. It is deliberately small and deterministic so
it can run as a PR evidence check.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from nsa.cognitive import NSACognitiveLM

PERTURBATIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def train_predictor(
    model: NSACognitiveLM,
    hidden: torch.Tensor,
    states: torch.Tensor,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
) -> float:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    for parameter in model.nsa.parameters():
        parameter.requires_grad_(False)

    predictor = model.self_model.predictor
    predictor.train()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)
    meaning = hidden[:, :-1].detach()
    previous_state = states[:, :-1].detach()
    target = states[:, 1:].detach()

    final_loss = 0.0
    for _ in range(epochs):
        prediction = model.self_model.predict(meaning, previous_state)
        loss = (prediction - target).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def rollout(
    model: NSACognitiveLM,
    tokens: torch.Tensor,
    initial_state: torch.Tensor,
    baseline_final: torch.Tensor,
    feedback: bool,
    steps: int,
) -> list[float]:
    state = initial_state
    distances: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            output = model(tokens, state_init=state, self_state_feedback=feedback)
            state = output["state"].detach()
            distance = (state[:, -1] - baseline_final).pow(2).mean().sqrt()
            distances.append(float(distance))
    return distances


def run(
    seed: int,
    *,
    epochs: int = 100,
    steps: int = 8,
    perturbations: list[float] | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    perturbations = PERTURBATIONS if perturbations is None else perturbations
    torch.manual_seed(seed)

    model = NSACognitiveLM(
        vocab_size=128,
        d_model=64,
        state_dim=8,
        num_layers=2,
        num_heads=4,
        max_seq_len=24,
        dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, 128, (4, 24))

    with torch.no_grad():
        _, hidden, native_state = model.nsa(tokens)
        hidden = hidden.detach()
        native_state = native_state.detach()

    training_loss = train_predictor(model, hidden, native_state, epochs=epochs)
    model.eval()

    rows: list[dict[str, Any]] = []
    for magnitude in perturbations:
        perturbation = torch.zeros_like(native_state)
        perturbation[..., 1:] = magnitude
        disturbed = native_state + perturbation
        feedback_on = rollout(model, tokens, disturbed, native_state[:, -1], True, steps)
        feedback_off = rollout(model, tokens, disturbed, native_state[:, -1], False, steps)
        initial = max(feedback_on[0], feedback_off[0], 1e-12)
        on_normalized = [value / initial for value in feedback_on]
        off_normalized = [value / initial for value in feedback_off]
        on_auc = sum(on_normalized)
        off_auc = sum(off_normalized)
        rows.append(
            {
                "perturbation": magnitude,
                "feedback_enabled": {
                    "normalized": on_normalized,
                    "final_normalized": on_normalized[-1],
                    "auc": on_auc,
                },
                "feedback_disabled": {
                    "normalized": off_normalized,
                    "final_normalized": off_normalized[-1],
                    "auc": off_auc,
                },
                "recovery_advantage": off_normalized[-1] - on_normalized[-1],
                "auc_advantage": off_auc - on_auc,
            }
        )

    recovery = [row["recovery_advantage"] for row in rows]
    auc = [row["auc_advantage"] for row in rows]
    return {
        "seed": seed,
        "epochs": epochs,
        "steps": steps,
        "training_loss": training_loss,
        "perturbations": perturbations,
        "results": rows,
        "summary": {
            "mean_recovery_advantage": sum(recovery) / len(recovery),
            "positive_recovery_advantage_fraction": sum(value > 0 for value in recovery) / len(recovery),
            "mean_auc_advantage": sum(auc) / len(auc),
            "positive_auc_advantage_fraction": sum(value > 0 for value in auc) / len(auc),
        },
        "finite": all(torch.isfinite(torch.tensor(value)).item() for value in [training_loss, *recovery, *auc]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, epochs=args.epochs, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
