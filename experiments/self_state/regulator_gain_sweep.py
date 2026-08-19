"""Sweep self-state regulator strength against trajectory-level recovery.

This experiment does not change the regulator defaults. It tests whether the
current contraction strength is too aggressive for the untrained model by
varying correction gain and maximum delta across deterministic seeds.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def rollout(model, tokens, state, target, enabled: bool, steps: int) -> list[float]:
    distances = []
    with torch.no_grad():
        for _ in range(steps):
            out = model(tokens, state_init=state, self_state_feedback=enabled)
            state = out["state"].detach()
            distances.append(float((state[:, -1] - target).pow(2).mean().sqrt()))
    return distances


def run(seed: int, gains: list[float], max_deltas: list[float],
        perturbations: list[float], steps: int = 8) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(
        vocab_size=128, d_model=64, state_dim=8, num_layers=2,
        num_heads=4, max_seq_len=24, dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, 128, (4, 24))
    with torch.no_grad():
        baseline = model(tokens, self_state_feedback=False)["base_state"].detach()

    rows = []
    for gain in gains:
        for max_delta in max_deltas:
            model.state_regulator.correction_gain = gain
            model.state_regulator.max_delta = max_delta
            for magnitude in perturbations:
                perturb = torch.zeros_like(baseline)
                perturb[..., 1:] = magnitude
                disturbed = baseline + perturb
                on = rollout(model, tokens, disturbed, baseline[:, -1], True, steps)
                off = rollout(model, tokens, disturbed, baseline[:, -1], False, steps)
                initial = max(on[0], off[0], 1e-12)
                on_norm = [x / initial for x in on]
                off_norm = [x / initial for x in off]
                on_auc = sum(on_norm)
                off_auc = sum(off_norm)
                rows.append({
                    "gain": gain,
                    "max_delta": max_delta,
                    "perturbation": magnitude,
                    "recovery_advantage": off_norm[-1] - on_norm[-1],
                    "auc_advantage": off_auc - on_auc,
                    "final_normalized": on_norm[-1],
                    "auc": on_auc,
                    "security_immutable": bool(torch.equal(
                        model(tokens, state_init=disturbed, self_state_feedback=True)["state"][..., 0],
                        disturbed[..., 0],
                    )),
                })

    finite = all(
        torch.isfinite(torch.tensor([
            r["gain"], r["max_delta"], r["perturbation"],
            r["recovery_advantage"], r["auc_advantage"],
            r["final_normalized"], r["auc"],
        ])).all().item()
        for r in rows
    )
    return {
        "seed": seed,
        "gains": gains,
        "max_deltas": max_deltas,
        "perturbations": perturbations,
        "steps": steps,
        "results": rows,
        "summary": {
            "mean_recovery_advantage": sum(r["recovery_advantage"] for r in rows) / len(rows),
            "mean_auc_advantage": sum(r["auc_advantage"] for r in rows) / len(rows),
            "positive_auc_advantage_fraction": sum(r["auc_advantage"] > 0 for r in rows) / len(rows),
            "all_security_immutable": all(r["security_immutable"] for r in rows),
        },
        "finite": finite,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gains", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--max-deltas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--perturbations", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.gains, args.max_deltas, args.perturbations, args.steps), indent=2))


if __name__ == "__main__":
    main()
