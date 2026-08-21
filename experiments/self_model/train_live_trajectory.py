"""Train and evaluate the Phase 19 predictive self-model on live trajectories.

The evaluator compares the learned predictor with a persistence baseline on the
same held-out transitions. This is intentionally a state-prediction experiment,
not a claim that the model has introspective access to transformer internals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nsa.predictive_self_model import PredictiveSelfModel, SELF_STATE_FIELDS


def _rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _tensor(row: dict, key: str) -> list[float]:
    return [float(row[key][field]) for field in SELF_STATE_FIELDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/live-self-model-training.json")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    rows = _rows(args.trajectory)
    if len(rows) < 2:
        raise ValueError("at least two trajectory rows are required for train/test evaluation")

    torch.manual_seed(args.seed)
    split = max(1, int(len(rows) * 0.8))
    train_rows, test_rows = rows[:split], rows[split:]
    if not test_rows:
        test_rows = train_rows[-1:]
        train_rows = train_rows[:-1] or train_rows

    model = PredictiveSelfModel(action_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_state = torch.tensor([_tensor(row, "state_before") for row in train_rows], dtype=torch.float32)
    train_action = torch.tensor([row["action"] if isinstance(row["action"], list) else [row["action"][name] for name in ("prompt_load", "max_tokens", "temperature", "output_load")] for row in train_rows], dtype=torch.float32)
    train_target = torch.tensor([_tensor(row, "state_after") for row in train_rows], dtype=torch.float32)

    losses: list[float] = []
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = model.training_loss(train_state, train_target, train_action)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    test_state = torch.tensor([_tensor(row, "state_before") for row in test_rows], dtype=torch.float32)
    test_action = torch.tensor([row["action"] if isinstance(row["action"], list) else [row["action"][name] for name in ("prompt_load", "max_tokens", "temperature", "output_load")] for row in test_rows], dtype=torch.float32)
    test_target = torch.tensor([_tensor(row, "state_after") for row in test_rows], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(test_state, test_action)
        predictor_mse = float(torch.mean((prediction - test_target) ** 2))
        persistence_mse = float(torch.mean((test_state - test_target) ** 2))

    result = {
        "seed": args.seed,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "epochs": args.epochs,
        "final_train_mse": losses[-1],
        "test_predictor_mse": predictor_mse,
        "test_persistence_mse": persistence_mse,
        "test_mse_improvement": persistence_mse - predictor_mse,
        "predictor_beats_persistence": predictor_mse < persistence_mse,
        "finite": all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
        "scientific_boundary": "explicit NSA self-state only; no transformer hidden-state access",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
