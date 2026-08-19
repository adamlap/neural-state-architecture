"""Causal self-state ablation experiment for the native NSA cognitive model."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 0, batch: int = 4, seq_len: int = 16, vocab_size: int = 128) -> dict[str, float]:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=vocab_size, d_model=64, state_dim=8, num_layers=2, num_heads=4, max_seq_len=seq_len, dropout=0.0)
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    targets = torch.roll(tokens, shifts=-1, dims=1)
    enabled = model(tokens, self_state_feedback=True)
    disabled = model(tokens, self_state_feedback=False)
    enabled_loss = F.cross_entropy(enabled["logits"].reshape(-1, vocab_size), targets.reshape(-1))
    disabled_loss = F.cross_entropy(disabled["logits"].reshape(-1, vocab_size), targets.reshape(-1))
    return {
        "enabled_loss": float(enabled_loss),
        "disabled_loss": float(disabled_loss),
        "logit_delta": float((enabled["logits"] - disabled["logits"]).abs().mean()),
        "prediction_mse": float(enabled["prediction_mse"].mean()),
        "caution_mean": float(enabled["caution"].mean()),
        "capability_mean": float(enabled["capability"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
