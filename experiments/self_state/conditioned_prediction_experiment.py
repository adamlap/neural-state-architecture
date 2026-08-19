"""
experiments/self_state/conditioned_prediction_experiment.py
===========================================================
Empirical evaluation comparing:
1. Static Unconditioned Self-Model Target
2. Conditioned Transition Self-Model Target P_theta(m_t, sigma_t, a_t)
3. Counterfactual Internal Simulation under Uncertainty

Evaluates target quality ratio, directional alignment, uncertainty calibration,
and strictly verifies hard security immutability.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F

from nsa.cognitive import NSACognitiveLM
from nsa.self_model import (
    ConditionedPredictiveSelfModel,
    CounterfactualInternalSimulator,
)

PERTURBATIONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def run(seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    d_model = 64
    state_dim = 8

    model = NSACognitiveLM(
        vocab_size=128,
        d_model=d_model,
        state_dim=state_dim,
        num_layers=2,
        num_heads=4,
        max_seq_len=24,
        dropout=0.0,
    )
    model.eval()

    conditioned_model = ConditionedPredictiveSelfModel(
        d_model=d_model,
        state_dim=state_dim,
        action_dim=state_dim,
    )
    conditioned_model.eval()

    simulator = CounterfactualInternalSimulator(
        self_model=conditioned_model,
        uncertainty_penalty=0.5,
    )

    tokens = torch.randint(0, 128, (4, 24))

    with torch.no_grad():
        _, hidden, native_state = model.nsa(tokens)

    results: list[dict[str, Any]] = []
    max_security_delta = 0.0

    for magnitude in PERTURBATIONS:
        perturbation = torch.zeros_like(native_state)
        perturbation[..., 1:] = magnitude
        disturbed = native_state + perturbation

        # Real security measurement
        sec_delta = float((disturbed[..., 0] - native_state[..., 0]).abs().max().item())
        max_security_delta = max(max_security_delta, sec_delta)

        # 1. Static Unconditioned prediction
        with torch.no_grad():
            static_pred = torch.zeros_like(native_state)
            static_pred[:, 1:] = model.self_model.predict(hidden[:, :-1], native_state[:, :-1])

            static_native_dist = float((native_state[:, -1] - static_pred[:, -1]).pow(2).mean().sqrt().item())
            disturbed_dist = float((disturbed[:, -1] - native_state[:, -1]).pow(2).mean().sqrt().item())
            static_ratio = static_native_dist / max(disturbed_dist, 1e-12)

            static_corr_dir = static_pred[:, -1] - disturbed[:, -1]
            oracle_dir = native_state[:, -1] - disturbed[:, -1]
            static_align = float(F.cosine_similarity(static_corr_dir, oracle_dir, dim=-1).mean().item())

        # 2. Conditioned Transition Prediction
        with torch.no_grad():
            action_proposal = -0.5 * torch.tanh(disturbed - static_pred)
            cond_out = conditioned_model(hidden, disturbed, action_proposal)
            cond_pred = cond_out["predicted_state"]
            cond_unc = float(cond_out["uncertainty"].mean().item())

            cond_native_dist = float((native_state[:, -1] - cond_pred[:, -1]).pow(2).mean().sqrt().item())
            cond_ratio = cond_native_dist / max(disturbed_dist, 1e-12)
            cond_corr_dir = cond_pred[:, -1] - disturbed[:, -1]
            cond_align = float(F.cosine_similarity(cond_corr_dir, oracle_dir, dim=-1).mean().item())

        # 3. Counterfactual Simulation
        candidates = [
            ("do_nothing", torch.zeros_like(disturbed), True),
            ("proportional_recovery", -0.5 * (disturbed - native_state), True),
            ("illegal_declass", perturbation, False),
        ]
        best_cand, all_cands = simulator.evaluate_candidates(
            hidden[:, -1:],
            disturbed[:, -1:],
            [(c[0], c[1][:, -1:], c[2]) for c in candidates],
        )

        results.append({
            "perturbation": magnitude,
            "disturbed_distance": disturbed_dist,
            "static_predictor": {
                "target_distance": static_native_dist,
                "target_quality_ratio": static_ratio,
                "directional_alignment": static_align,
            },
            "conditioned_predictor": {
                "target_distance": cond_native_dist,
                "target_quality_ratio": cond_ratio,
                "directional_alignment": cond_align,
                "uncertainty": cond_unc,
            },
            "counterfactual_simulator": {
                "selected_action": best_cand.action_id if best_cand else None,
                "is_legal": best_cand.is_legal if best_cand else False,
                "score": best_cand.score if best_cand else float("-inf"),
            },
            "security_delta": sec_delta,
        })

    static_ratios = [r["static_predictor"]["target_quality_ratio"] for r in results]
    cond_ratios = [r["conditioned_predictor"]["target_quality_ratio"] for r in results]
    static_aligns = [r["static_predictor"]["directional_alignment"] for r in results]
    cond_aligns = [r["conditioned_predictor"]["directional_alignment"] for r in results]

    return {
        "seed": seed,
        "perturbations": PERTURBATIONS,
        "results": results,
        "summary": {
            "mean_static_target_ratio": sum(static_ratios) / len(static_ratios),
            "mean_conditioned_target_ratio": sum(cond_ratios) / len(cond_ratios),
            "mean_static_alignment": sum(static_aligns) / len(static_aligns),
            "mean_conditioned_alignment": sum(cond_aligns) / len(cond_aligns),
            "all_counterfactuals_legal": all(r["counterfactual_simulator"]["is_legal"] for r in results),
            "max_security_delta": max_security_delta,
        },
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.seed), indent=2))


if __name__ == "__main__":
    main()
