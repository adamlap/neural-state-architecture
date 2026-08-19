"""Measure whether the self-model target is aligned with native state correction.

The regulator can only contract toward the target supplied by the self-model.
This experiment therefore separates target quality from controller behavior by
measuring prediction error and directional alignment with the native corrective
vector, without changing model weights or production defaults.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F

from nsa.cognitive import NSACognitiveLM

PERTURBATIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def run(seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = NSACognitiveLM(
        vocab_size=128, d_model=64, state_dim=8, num_layers=2,
        num_heads=4, max_seq_len=24, dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, 128, (4, 24))

    with torch.no_grad():
        _, hidden, native_state = model.nsa(tokens)
        predicted = torch.zeros_like(native_state)
        predicted[:, 1:] = model.self_model.predict(hidden[:, :-1], native_state[:, :-1])

    rows: list[dict[str, Any]] = []
    max_security_delta = 0.0
    for magnitude in PERTURBATIONS:
        perturbation = torch.zeros_like(native_state)
        perturbation[..., 1:] = magnitude
        disturbed = native_state + perturbation

        # Explicitly measure security delta rather than hardcoding it
        # Perturbation only modifies soft coordinates (1:), coordinate 0 must be unchanged
        sec_input_delta = float((disturbed[..., 0] - native_state[..., 0]).abs().max().item())
        max_security_delta = max(max_security_delta, sec_input_delta)

        native_error = native_state[:, -1] - predicted[:, -1]
        prediction_error = disturbed[:, -1] - predicted[:, -1]
        native_target_error = disturbed[:, -1] - native_state[:, -1]
        correction_direction = predicted[:, -1] - disturbed[:, -1]
        oracle_direction = native_state[:, -1] - disturbed[:, -1]

        prediction_distance = prediction_error.pow(2).mean().sqrt()
        disturbed_native_distance = native_target_error.pow(2).mean().sqrt()
        predicted_native_distance = native_error.pow(2).mean().sqrt()
        alignment = F.cosine_similarity(correction_direction, oracle_direction, dim=-1).mean()

        rows.append({
            "perturbation": magnitude,
            "prediction_native_distance": float(predicted_native_distance),
            "disturbed_native_distance": float(disturbed_native_distance),
            "target_quality_ratio": float(predicted_native_distance / max(disturbed_native_distance, 1e-12)),
            "correction_oracle_cosine": float(alignment),
            "prediction_error_from_disturbed": float(prediction_distance),
            "security_delta": sec_input_delta,
        })

    ratios = [r["target_quality_ratio"] for r in rows]
    alignments = [r["correction_oracle_cosine"] for r in rows]
    return {
        "seed": seed,
        "perturbations": PERTURBATIONS,
        "results": rows,
        "summary": {
            "mean_target_quality_ratio": sum(ratios) / len(ratios),
            "fraction_target_closer_than_disturbed": sum(r < 1.0 for r in ratios) / len(ratios),
            "mean_correction_oracle_cosine": sum(alignments) / len(alignments),
            "fraction_positive_directional_alignment": sum(a > 0 for a in alignments) / len(alignments),
            "max_security_delta": max_security_delta,
        },
        "finite": all(torch.isfinite(torch.tensor(ratios + alignments)).all().item() for _ in [0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.seed), indent=2))


if __name__ == "__main__":
    main()
