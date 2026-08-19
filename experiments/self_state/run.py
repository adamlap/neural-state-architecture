"""Run the first NSA explicit-self-state experiment.

Example:
    PYTHONPATH=. python experiments/self_state/run.py --steps 800 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from experiments.self_state.model import BaselineEvidenceModel, ExplicitSelfStateModel, parameter_count
from experiments.self_state.task import make_batch, make_shifted_batch


@dataclass
class Metrics:
    accuracy: float
    brier: float
    ece: float
    selective_accuracy: float
    coverage: float
    mean_confidence: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def ece(prob: Tensor, target: Tensor, bins: int = 10) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=prob.device)
    total = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob <= edges[i + 1] if i == bins - 1 else prob < edges[i + 1])
        if mask.any():
            total += float(mask.float().mean()) * abs(
                float(prob[mask].mean()) - float(target[mask].mean())
            )
    return total


def evaluate(model: nn.Module, x: Tensor, y: Tensor, state_scale: float = 1.0) -> Metrics:
    model.eval()
    with torch.no_grad():
        out = model(x, state_scale=state_scale) if isinstance(model, ExplicitSelfStateModel) else model(x)
        prob = torch.sigmoid(out["logits"])
        confidence = out["confidence"]
        target = y.float()
        pred = (prob >= 0.5).float()
        accuracy = float((pred == target).float().mean())
        brier = float(((prob - target) ** 2).mean())
        calibration_target = (pred == target).float()
        calibration_prob = torch.where(pred > 0.5, confidence, 1.0 - confidence)
        calibration_prob = calibration_prob.clamp(0.0, 1.0)
        calibration_ece = ece(calibration_prob, calibration_target)
        selected = confidence >= 0.7
        coverage = float(selected.float().mean())
        selective_accuracy = float((pred[selected] == target[selected]).float().mean()) if selected.any() else 0.0
        return Metrics(
            accuracy=accuracy,
            brier=brier,
            ece=calibration_ece,
            selective_accuracy=selective_accuracy,
            coverage=coverage,
            mean_confidence=float(confidence.mean()),
        )


def train_baseline(model: BaselineEvidenceModel, steps: int, batch_size: int, lr: float, seed: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + 1000)
    model.train()
    for _ in range(steps):
        x, y = make_batch(batch_size, generator=generator)
        out = model(x)
        target_confidence = (torch.sigmoid(out["logits"]).detach().round() == y).float()
        loss = loss_fn(out["logits"], y) + 0.25 * ((out["confidence"] - target_confidence) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_self_state(model: ExplicitSelfStateModel, steps: int, batch_size: int, lr: float, seed: int) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed + 2000)
    model.train()
    for _ in range(steps):
        x, y = make_batch(batch_size, generator=generator)
        out = model(x)
        prob = torch.sigmoid(out["logits"])
        correctness = (prob.detach().round() == y).float()
        state = out["self_state"]
        # Explicit epistemic targets: confidence and uncertainty are trained
        # against objective correctness. Other dimensions remain latent.
        state_loss = 0.5 * ((state[:, 0] - correctness) ** 2).mean()
        state_loss += 0.5 * ((state[:, 1] - (1.0 - correctness)) ** 2).mean()
        loss = bce(out["logits"], y) + 0.25 * ((out["confidence"] - correctness) ** 2).mean() + state_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run(seed: int, steps: int, batch_size: int, lr: float) -> dict[str, object]:
    set_seed(seed)
    baseline = BaselineEvidenceModel(hidden=32)
    explicit = ExplicitSelfStateModel(hidden=28)

    train_baseline(baseline, steps, batch_size, lr, seed)
    train_self_state(explicit, steps, batch_size, lr, seed)

    test_gen = torch.Generator().manual_seed(seed + 3000)
    x, y = make_batch(4096, generator=test_gen)
    shifted_x, shifted_y = make_shifted_batch(4096, generator=torch.Generator().manual_seed(seed + 4000))

    baseline_iid = evaluate(baseline, x, y)
    explicit_iid = evaluate(explicit, x, y)
    baseline_shift = evaluate(baseline, shifted_x, shifted_y)
    explicit_shift = evaluate(explicit, shifted_x, shifted_y)
    explicit_ablation = evaluate(explicit, x, y, state_scale=0.0)

    return {
        "seed": seed,
        "steps": steps,
        "parameter_count": {
            "baseline": parameter_count(baseline),
            "explicit_self_state": parameter_count(explicit),
        },
        "iid": {
            "baseline": baseline_iid.as_dict(),
            "explicit_self_state": explicit_iid.as_dict(),
        },
        "shifted": {
            "baseline": baseline_shift.as_dict(),
            "explicit_self_state": explicit_shift.as_dict(),
        },
        "causal_ablation": {
            "explicit_self_state": explicit_ablation.as_dict(),
            "state_path_delta_accuracy": explicit_iid.accuracy - explicit_ablation.accuracy,
            "state_path_delta_brier": explicit_ablation.brier - explicit_iid.brier,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()
    result = run(args.seed, args.steps, args.batch_size, args.lr)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
