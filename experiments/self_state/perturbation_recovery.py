"""State perturbation/recovery experiment for NSA self-monitoring."""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 7, batch: int = 2, seq_len: int = 12, vocab_size: int = 128,
        perturbation: float = 2.0) -> dict[str, float | list[int]]:
    torch.manual_seed(seed)
    model = NSACognitiveLM(
        vocab_size=vocab_size, d_model=64, state_dim=8,
        num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        normal = model(tokens, self_state_feedback=True)
        disturbed_init = torch.zeros_like(normal["state"])
        disturbed_init[:, seq_len // 2 :, :] = perturbation
        disturbed = model(tokens, state_init=disturbed_init, self_state_feedback=True)

    normal_mse = float(normal["prediction_mse"].mean())
    disturbed_mse = float(disturbed["prediction_mse"].mean())
    return {
        "logits_shape": list(normal["logits"].shape),
        "state_shape": list(normal["state"].shape),
        "normal_prediction_mse": normal_mse,
        "disturbed_prediction_mse": disturbed_mse,
        "prediction_error_increase": disturbed_mse - normal_mse,
        "state_recovery_error": float((normal["state"] - disturbed["state"]).abs().mean()),
        "logit_shift": float((normal["logits"] - disturbed["logits"]).abs().mean()),
        "caution_shift": float(disturbed["caution"].mean() - normal["caution"].mean()),
        "perturbation": perturbation,
        "finite": float(bool(torch.isfinite(disturbed["logits"]).all() and torch.isfinite(disturbed["state"]).all())),
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
