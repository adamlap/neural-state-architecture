"""Measure one-step self-state correction without recursive rollout.

The recursive regulation experiments show that feedback can improve or worsen
long-horizon trajectories. This experiment isolates the local operation:
does one bounded regulator update move a perturbed state toward the native
state, and does it reduce prediction error? No model weights are changed.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from nsa.cognitive import NSACognitiveLM

PERTURBATIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def run(seed: int, *, correction_gain: float = 0.5, max_delta: float = 0.25) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = NSACognitiveLM(
        vocab_size=128, d_model=64, state_dim=8, num_layers=2,
        num_heads=4, max_seq_len=24, dropout=0.0,
    )
    model.eval()
    model.state_regulator.correction_gain = correction_gain
    model.state_regulator.max_delta = max_delta
    tokens = torch.randint(0, 128, (4, 24))

    with torch.no_grad():
        _, hidden, native_state = model.nsa(tokens)
        predicted = torch.zeros_like(native_state)
        predicted[:, 1:] = model.self_model.predict(hidden[:, :-1], native_state[:, :-1])
        native_final = native_state[:, -1]
        predicted_final = predicted[:, -1]

    rows: list[dict[str, Any]] = []
    for magnitude in PERTURBATIONS:
        perturbation = torch.zeros_like(native_state)
        perturbation[..., 1:] = magnitude
        disturbed = native_state + perturbation
        with torch.no_grad():
            prediction_error = disturbed - predicted
            regulated = model.state_regulator(disturbed, prediction_error, enabled=True)
            before_native = (disturbed[:, -1] - native_final).pow(2).mean().sqrt()
            after_native = (regulated[:, -1] - native_final).pow(2).mean().sqrt()
            before_prediction = (disturbed[:, -1] - predicted_final).pow(2).mean().sqrt()
            after_prediction = (regulated[:, -1] - predicted_final).pow(2).mean().sqrt()
            delta = (regulated - disturbed).abs()[..., 1:].max()
            security_delta = (regulated[..., 0] - disturbed[..., 0]).abs().max()
        rows.append({
            "perturbation": magnitude,
            "native_distance_before": float(before_native),
            "native_distance_after": float(after_native),
            "native_contraction": float((before_native - after_native) / max(before_native, 1e-12)),
            "prediction_distance_before": float(before_prediction),
            "prediction_distance_after": float(after_prediction),
            "prediction_contraction": float((before_prediction - after_prediction) / max(before_prediction, 1e-12)),
            "max_soft_delta": float(delta),
            "security_delta": float(security_delta),
        })

    native = [r["native_contraction"] for r in rows]
    prediction = [r["prediction_contraction"] for r in rows]
    return {
        "seed": seed,
        "correction_gain": correction_gain,
        "max_delta": max_delta,
        "perturbations": PERTURBATIONS,
        "results": rows,
        "summary": {
            "mean_native_contraction": sum(native) / len(native),
            "positive_native_contraction_fraction": sum(x > 0 for x in native) / len(native),
            "mean_prediction_contraction": sum(prediction) / len(prediction),
            "positive_prediction_contraction_fraction": sum(x > 0 for x in prediction) / len(prediction),
            "max_security_delta": max(r["security_delta"] for r in rows),
        },
        "finite": all(torch.isfinite(torch.tensor(x)).item() for x in [*native, *prediction]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correction-gain", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, correction_gain=args.correction_gain, max_delta=args.max_delta), indent=2))


if __name__ == "__main__":
    main()
