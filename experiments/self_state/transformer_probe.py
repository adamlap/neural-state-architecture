"""Small executable probe against the repository's native NSACausalLM.

This is an integration smoke test, not a claim of benchmark superiority.
It verifies that semantic computation and structured state are both active,
that state initialization changes the computation, and that outputs remain
finite and shape-correct.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.layers import NSACausalLM


def run(batch: int = 2, seq_len: int = 16, vocab_size: int = 128, state_dim: int = 8) -> dict[str, float | list[int]]:
    torch.manual_seed(0)
    model = NSACausalLM(
        vocab_size=vocab_size,
        d_model=64,
        state_dim=state_dim,
        num_layers=2,
        num_heads=4,
        max_seq_len=seq_len,
        dropout=0.0,
    )
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    logits, hidden, state = model(tokens)
    zero_state = torch.zeros_like(state)
    logits_zero, _, state_zero = model(tokens, state_init=zero_state)
    delta = (logits - logits_zero).abs().mean()
    return {
        "logits_shape": list(logits.shape),
        "state_shape": list(state.shape),
        "hidden_shape": list(hidden.shape),
        "state_norm": float(state.norm()),
        "state_zero_norm": float(state_zero.norm()),
        "logit_delta_from_zero_init": float(delta),
        "finite": float(bool(torch.isfinite(logits).all() and torch.isfinite(state).all())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(batch=args.batch, seq_len=args.seq_len), indent=2))


if __name__ == "__main__":
    main()
