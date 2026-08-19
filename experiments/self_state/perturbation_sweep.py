"""Multi-seed perturbation sweep for NSA self-state regulation.

Reports normalized recovery and area-under-the-distance-curve (AUC) so the
metric remains comparable across perturbation magnitudes. This is deliberately
an untrained architectural experiment: it tests connectivity and stability,
not learned intelligence or safety performance.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def _rollout(model, tokens, initial_state, baseline_final, feedback, steps):
    state = initial_state
    distances = []
    with torch.no_grad():
        for _ in range(steps):
            out = model(tokens, state_init=state, self_state_feedback=feedback)
            state = out["state"].detach()
            distances.append(float((state[:, -1] - baseline_final).pow(2).mean().sqrt()))
    return distances


def _finite(value: float) -> bool:
    """Return a real Python bool for a scalar experiment metric."""
    return bool(torch.isfinite(torch.tensor(value)).item())


def run(seed: int, perturbations: list[float], steps: int = 8, batch: int = 4,
        seq_len: int = 24, vocab_size: int = 128) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=vocab_size, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        baseline = model(tokens, self_state_feedback=False)["base_state"].detach()
    rows = []
    for magnitude in perturbations:
        perturb = torch.zeros_like(baseline)
        perturb[..., 1:] = magnitude
        disturbed = baseline + perturb
        on = _rollout(model, tokens, disturbed, baseline[:, -1], True, steps)
        off = _rollout(model, tokens, disturbed, baseline[:, -1], False, steps)
        initial = max(on[0], off[0], 1e-12)
        on_norm = [x / initial for x in on]
        off_norm = [x / initial for x in off]
        on_auc = sum(on_norm)
        off_auc = sum(off_norm)
        rows.append({
            "perturbation": magnitude,
            "initial_distance": initial,
            "feedback_enabled": {"distances": on, "normalized": on_norm,
                                  "final_normalized": on_norm[-1], "auc": on_auc,
                                  "recovery_fraction": 1.0 - on_norm[-1]},
            "feedback_disabled": {"distances": off, "normalized": off_norm,
                                   "final_normalized": off_norm[-1], "auc": off_auc,
                                   "recovery_fraction": 1.0 - off_norm[-1]},
            "recovery_advantage": off_norm[-1] - on_norm[-1],
            "auc_advantage": off_auc - on_auc,
        })

    metrics = []
    for row in rows:
        metrics.extend([
            row["initial_distance"],
            *row["feedback_enabled"]["distances"],
            *row["feedback_enabled"]["normalized"],
            row["feedback_enabled"]["auc"],
            row["feedback_enabled"]["final_normalized"],
            *row["feedback_disabled"]["distances"],
            *row["feedback_disabled"]["normalized"],
            row["feedback_disabled"]["auc"],
            row["feedback_disabled"]["final_normalized"],
            row["recovery_advantage"],
            row["auc_advantage"],
        ])
    finite = all(_finite(value) for value in metrics)
    if not finite:
        raise RuntimeError("Self-state perturbation sweep produced NaN or Inf metrics")

    advantages = [r["recovery_advantage"] for r in rows]
    auc_advantages = [r["auc_advantage"] for r in rows]
    return {
        "seed": seed,
        "steps": steps,
        "perturbations": perturbations,
        "results": rows,
        "summary": {
            "mean_recovery_advantage": sum(advantages) / len(advantages),
            "positive_recovery_advantage_fraction": sum(a > 0 for a in advantages) / len(advantages),
            "mean_auc_advantage": sum(auc_advantages) / len(auc_advantages),
        },
        "finite": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--perturbations", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    args = p.parse_args()
    print(json.dumps(run(args.seed, args.perturbations, args.steps), indent=2))


if __name__ == "__main__":
    main()
