"""State perturbation experiment with explicit feedback ablation and recovery metrics."""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 7, batch: int = 2, seq_len: int = 12, vocab_size: int = 128,
        perturbation: float = 2.0, recovery_steps: int = 6, epsilon: float = 0.25):
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=vocab_size, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        baseline = model(tokens, self_state_feedback=True)
        baseline_state = baseline["state"]
        baseline_final = baseline_state[:, -1]
        baseline_mse = float(baseline["prediction_mse"].mean())
        baseline_caution = float(baseline["caution"].mean())

        results = {}
        for feedback in (True, False):
            distances, mses, cautions, logit_shifts = [], [], [], []
            for step in range(recovery_steps + 1):
                # At step 0 the full disturbance is present; it is gradually
                # removed so the experiment measures a trajectory rather than
                # conflating disturbance magnitude with endpoint error.
                scale = perturbation * max(0.0, 1.0 - step / max(1, recovery_steps))
                init = torch.zeros_like(baseline_state)
                init[:, seq_len // 2 :, :] = scale
                out = model(tokens, state_init=init, self_state_feedback=feedback)
                final_state = out["state"][:, -1]
                distances.append(float((final_state - baseline_final).pow(2).mean().sqrt()))
                mses.append(float(out["prediction_mse"].mean()))
                cautions.append(float(out["caution"].mean()))
                logit_shifts.append(float((out["logits"] - baseline["logits"]).abs().mean()))

            recovery_step = next((i for i, d in enumerate(distances) if d <= epsilon), None)
            results["feedback_enabled" if feedback else "feedback_disabled"] = {
                "distances": distances,
                "prediction_mse": mses,
                "caution": cautions,
                "logit_shift": logit_shifts,
                "initial_distance": distances[0],
                "final_distance": distances[-1],
                "recovery_step": recovery_step,
                "mse_delta_initial": mses[0] - baseline_mse,
                "caution_delta_initial": cautions[0] - baseline_caution,
            }

    enabled = results["feedback_enabled"]
    disabled = results["feedback_disabled"]
    return {
        "logits_shape": list(baseline["logits"].shape),
        "state_shape": list(baseline["state"].shape),
        "baseline_prediction_mse": baseline_mse,
        "baseline_caution": baseline_caution,
        "perturbation": perturbation,
        "epsilon": epsilon,
        "recovery_steps": recovery_steps,
        "results": results,
        "feedback_recovery_advantage": (
            disabled["final_distance"] - enabled["final_distance"]
        ),
        "feedback_mse_delta_advantage": (
            disabled["mse_delta_initial"] - enabled["mse_delta_initial"]
        ),
        "finite": float(all(
            torch.isfinite(torch.tensor(v)).all()
            for branch in results.values()
            for values in branch.values()
            if isinstance(values, (int, float))
            for v in [values]
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--perturbation", type=float, default=2.0)
    parser.add_argument("--recovery-steps", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, seq_len=args.seq_len,
                         perturbation=args.perturbation,
                         recovery_steps=args.recovery_steps,
                         epsilon=args.epsilon), indent=2))


if __name__ == "__main__":
    main()
