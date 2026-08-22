"""Cross-seed and cross-dynamics validation for CCE state prediction.

This is a research gate, not a consciousness claim. It asks whether the
learned one-step predictor from ``nsa.runtime.predictive_dynamics`` beats the
trivial persistence predictor on held-out trajectory transitions across
multiple random seeds and dynamical families.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from nsa.runtime.predictive_dynamics import StatePredictor, prediction_metrics, train_predictor


@dataclass(frozen=True)
class Task:
    name: str
    a: float
    b: float
    c: float
    bias: float


TASKS = (
    Task("stable_linear", 0.72, 0.00, 0.18, 0.05),
    Task("nonlinear_damped", 0.63, 0.12, 0.16, 0.03),
    Task("weakly_forced", 0.81, -0.07, 0.22, -0.02),
)


def make_trajectory(task: Task, seed: int, length: int = 96) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state = torch.empty(1).uniform_(-1.0, 1.0, generator=generator)
    states: list[torch.Tensor] = []
    external: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for step in range(length):
        u = torch.tensor([0.6 * torch.sin(torch.tensor(step * 0.17 + seed * 0.11))])
        noise = 0.002 * torch.randn(1, generator=generator)
        target = task.a * state + task.b * torch.tanh(state) + task.c * u + task.bias + noise
        states.append(state.clone())
        external.append(u)
        targets.append(target.clone())
        state = target
    return torch.stack(states), torch.stack(external), torch.stack(targets)


def run(seed: int, epochs: int) -> dict[str, object]:
    torch.manual_seed(seed)
    task_rows: list[dict[str, object]] = []
    for task_index, task in enumerate(TASKS):
        states, external, targets = make_trajectory(task, seed * 100 + task_index)
        split = int(len(states) * 0.75)
        model = StatePredictor(state_dim=1, input_dim=1, hidden_dim=24)
        train_predictor(
            model,
            states[:split],
            targets[:split],
            external[:split],
            epochs=epochs,
            learning_rate=5e-3,
        )
        model.eval()
        with torch.no_grad():
            predicted = model(states[split:], external[split:])
        metrics = prediction_metrics(predicted, targets[split:], states[split:])
        task_rows.append({
            "task": asdict(task),
            "evaluation_transitions": len(states) - split,
            "mse": metrics.mse,
            "persistence_mse": metrics.persistence_mse,
            "improvement": metrics.improvement,
            "beats_persistence": metrics.mse < metrics.persistence_mse,
        })
    return {
        "seed": seed,
        "epochs": epochs,
        "tasks": task_rows,
        "all_tasks_beat_persistence": all(bool(row["beats_persistence"]) for row in task_rows),
        "mean_improvement": sum(float(row["improvement"]) for row in task_rows) / len(task_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 42, 73, 101])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = [run(seed, args.epochs) for seed in args.seeds]
    seed_success = [bool(item["all_tasks_beat_persistence"]) for item in runs]
    result = {
        "experiment": "cce_predictive_multiseed",
        "claim_boundary": "held-out transition-learning benchmark; no consciousness claim",
        "seeds": args.seeds,
        "runs": runs,
        "successful_seeds": sum(seed_success),
        "required_successful_seeds": len(args.seeds),
        "all_seeds_all_tasks_beat_persistence": all(seed_success),
        "mean_seed_improvement": sum(float(item["mean_improvement"]) for item in runs) / len(runs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
