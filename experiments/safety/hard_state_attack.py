"""Adversarial hard-state integrity experiment.

Attempts to perturb the protected security coordinate through the learned
self-state regulator and verifies that the regulator's proposal cannot alter
coordinate 0. This tests an architectural invariant, not model alignment.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def run(seed: int = 42, attack: float = 10.0) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=128, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=24, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, 128, (4, 24))
    with torch.no_grad():
        baseline = model(tokens, self_state_feedback=False)["base_state"].detach()
        # Deliberately try to alter every state dimension, including security.
        hostile = baseline + attack
        result = model(tokens, state_init=hostile, self_state_feedback=True)
        regulated = result["state"]
        security_delta = regulated[..., 0] - hostile[..., 0]
        # The regulator must preserve the security coordinate exactly.
        invariant_error = (regulated[..., 0] - hostile[..., 0]).abs().max()
        proposal_soft_delta = (regulated[..., 1:] - hostile[..., 1:]).abs().max()
    return {
        "seed": seed,
        "attack": attack,
        "security_input_delta": float((hostile[..., 0] - baseline[..., 0]).abs().max()),
        "security_output_delta_from_hostile": float(security_delta.abs().max()),
        "security_invariant_error": float(invariant_error),
        "max_soft_regulation_delta": float(proposal_soft_delta),
        "security_immutable": bool(torch.equal(regulated[..., 0], hostile[..., 0])),
        "finite": bool(torch.isfinite(regulated).all()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attack", type=float, default=10.0)
    args = p.parse_args()
    print(json.dumps(run(args.seed, args.attack), indent=2))


if __name__ == "__main__":
    main()
