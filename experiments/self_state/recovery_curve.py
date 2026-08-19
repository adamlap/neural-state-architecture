"""Causal recovery benchmark for the closed NSA self-state loop."""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def _trajectory(model, tokens, initial_state, feedback, steps):
    state = initial_state
    distances, mse, caution = [], [], []
    with torch.no_grad():
        for _ in range(steps):
            out = model(tokens, state_init=state, self_state_feedback=feedback)
            state = out["state"].detach()
            distances.append(float(state[:, -1].pow(2).mean().sqrt()))
            mse.append(float(out["prediction_mse"].mean()))
            caution.append(float(out["caution"].mean()))
    return distances, mse, caution


def run(seed: int = 7, batch: int = 4, seq_len: int = 24, vocab_size: int = 128,
        perturbation: float = 2.0, recovery_steps: int = 6, epsilon: float = 0.25) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=vocab_size, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        baseline_out = model(tokens, self_state_feedback=False)
        baseline = baseline_out["base_state"].detach()
        perturb = torch.zeros_like(baseline)
        perturb[:, seq_len // 2 :, 1:] = perturbation
        disturbed_init = baseline + perturb

    # Compare the same disturbed state under the closed loop and its ablation.
    enabled_d, enabled_mse, enabled_caution = _trajectory(
        model, tokens, disturbed_init, True, recovery_steps
    )
    disabled_d, disabled_mse, disabled_caution = _trajectory(
        model, tokens, disturbed_init, False, recovery_steps
    )

    # Distances are measured against the unperturbed baseline trajectory at the
    # final recurrent state, not against zero, so this is a genuine recovery metric.
    def recovery_distances(feedback):
        state = disturbed_init
        values = []
        with torch.no_grad():
            for _ in range(recovery_steps):
                out = model(tokens, state_init=state, self_state_feedback=feedback)
                state = out["state"].detach()
                values.append(float((state[:, -1] - baseline[:, -1]).pow(2).mean().sqrt()))
        return values

    enabled_recovery = recovery_distances(True)
    disabled_recovery = recovery_distances(False)
    enabled_final = enabled_recovery[-1]
    disabled_final = disabled_recovery[-1]
    advantage = disabled_final - enabled_final
    return {
        "seed": seed,
        "perturbation": perturbation,
        "epsilon": epsilon,
        "recovery_steps": recovery_steps,
        "feedback_enabled": {
            "state_distance": enabled_recovery,
            "raw_state_norm": enabled_d,
            "prediction_mse": enabled_mse,
            "caution": enabled_caution,
            "final_distance": enabled_final,
            "recovery_step": next((i for i, d in enumerate(enabled_recovery) if d <= epsilon), None),
        },
        "feedback_disabled": {
            "state_distance": disabled_recovery,
            "raw_state_norm": disabled_d,
            "prediction_mse": disabled_mse,
            "caution": disabled_caution,
            "final_distance": disabled_final,
            "recovery_step": next((i for i, d in enumerate(disabled_recovery) if d <= epsilon), None),
        },
        "feedback_recovery_advantage": advantage,
        "finite": float(all(torch.isfinite(torch.tensor(x)) for x in enabled_recovery + disabled_recovery)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--perturbation", type=float, default=2.0)
    p.add_argument("--recovery-steps", type=int, default=6)
    p.add_argument("--epsilon", type=float, default=0.25)
    args = p.parse_args()
    print(json.dumps(run(args.seed, perturbation=args.perturbation,
                         recovery_steps=args.recovery_steps, epsilon=args.epsilon), indent=2))


if __name__ == "__main__":
    main()
