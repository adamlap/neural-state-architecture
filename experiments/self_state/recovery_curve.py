"""Measure recovery of NSA self-state after a transient internal disturbance."""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 7, batch: int = 4, seq_len: int = 24, vocab_size: int = 128,
        perturbation: float = 2.0, recovery_steps: int = 6, epsilon: float = 0.25) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=vocab_size, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        _, _, normal_state, _ = model(tokens, self_state_feedback=True)
        baseline = normal_state[:, -1]
        distances = []
        errors = []
        for step in range(recovery_steps + 1):
            scale = perturbation * max(0.0, 1.0 - step / max(1, recovery_steps))
            init = torch.zeros_like(normal_state)
            init[:, seq_len // 2 :, :] = scale
            out = model(tokens, state_init=init, self_state_feedback=True)
            final_state = out["state"][:, -1]
            distances.append(float((final_state - baseline).pow(2).mean().sqrt()))
            errors.append(float(out["prediction_mse"].mean()))
    recovery_time = next((i for i, d in enumerate(distances) if d <= epsilon), None)
    return {
        "seed": seed,
        "perturbation": perturbation,
        "epsilon": epsilon,
        "distances": distances,
        "prediction_mse": errors,
        "recovery_step": recovery_time,
        "final_distance": distances[-1],
        "finite": float(all(torch.isfinite(torch.tensor(x)) for x in distances + errors)),
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
