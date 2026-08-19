"""State perturbation/recovery experiment for NSA self-monitoring.

This experiment asks whether predictive self-state reacts to an artificial
internal-state disturbance. It reports metrics only; it makes no claim about
consciousness.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 7, batch: int = 2, seq_len: int = 12, vocab_size: int = 128, state_dim: int = 8,
        perturbation: float = 2.0) -> dict[str, float | list[int]]:
    torch.manual_seed(seed)
    model = NSACognitiveLM(
        vocab_size=vocab_size, d_model=64, state_dim=state_dim,
        num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        normal_logits, _, normal_state, normal_self = model(tokens, self_state_feedback=True)
        disturbed_init = torch.zeros_like(normal_state)
        disturbed_init[:, seq_len // 2 :, :] = perturbation
        disturbed_logits, _, disturbed_state, disturbed_self = model(
            tokens, state_init=disturbed_init, self_state_feedback=True
        )
        recovery = model.state_recovery_error(normal_state, disturbed_state)
    normal_error = float(normal_self.prediction_mse)
    disturbed_error = float(disturbed_self.prediction_mse)
    return {
        "logits_shape": list(normal_logits.shape),
        "state_shape": list(normal_state.shape),
        "normal_prediction_mse": normal_error,
        "disturbed_prediction_mse": disturbed_error,
        "prediction_error_increase": disturbed_error - normal_error,
        "state_recovery_error": float(recovery),
        "logit_shift": float((normal_logits - disturbed_logits).abs().mean()),
        "perturbation": perturbation,
        "finite": float(bool(torch.isfinite(disturbed_logits).all() and torch.isfinite(disturbed_state).all())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--perturbation", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed, seq_len=args.seq_len, perturbation=args.perturbation), indent=2))


if __name__ == "__main__":
    main()
