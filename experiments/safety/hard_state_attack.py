"""Adversarial hard-state integrity experiments.

The experiment suite attacks the protected security coordinate through several
entry points and verifies that the hard coordinate remains invariant while
soft dimensions remain available to regulation. This tests an architectural
invariant, not model alignment.
"""
from __future__ import annotations

import argparse
import json

import torch

from nsa.cognitive import NSACognitiveLM


def _case(model: NSACognitiveLM, tokens: torch.Tensor, baseline: torch.Tensor,
          attack: float, mode: str) -> dict:
    if mode == "state_init":
        hostile = baseline + attack
    elif mode == "security_only":
        hostile = baseline.clone()
        hostile[..., 0] += attack
    elif mode == "soft_only":
        hostile = baseline.clone()
        hostile[..., 1:] += attack
    else:
        raise ValueError(f"unknown attack mode: {mode}")

    with torch.no_grad():
        result = model(tokens, state_init=hostile, self_state_feedback=True)
        regulated = result["state"]

    security_delta = regulated[..., 0] - hostile[..., 0]
    invariant_error = security_delta.abs().max()
    soft_delta = (regulated[..., 1:] - hostile[..., 1:]).abs().max()
    return {
        "mode": mode,
        "attack": attack,
        "security_input_delta": float((hostile[..., 0] - baseline[..., 0]).abs().max()),
        "security_output_delta_from_hostile": float(security_delta.abs().max()),
        "security_invariant_error": float(invariant_error),
        "max_soft_regulation_delta": float(soft_delta),
        "security_immutable": bool(torch.equal(regulated[..., 0], hostile[..., 0])),
        "finite": bool(torch.isfinite(regulated).all().item()),
    }


def run(seed: int = 42, attack: float = 10.0,
        modes: tuple[str, ...] = ("state_init", "security_only", "soft_only")) -> dict:
    torch.manual_seed(seed)
    model = NSACognitiveLM(vocab_size=128, d_model=64, state_dim=8,
                           num_layers=2, num_heads=4, max_seq_len=24, dropout=0.0)
    model.eval()
    tokens = torch.randint(0, 128, (4, 24))
    with torch.no_grad():
        baseline = model(tokens, self_state_feedback=False)["base_state"].detach()
    cases = [_case(model, tokens, baseline, attack, mode) for mode in modes]
    return {
        "seed": seed,
        "attack": attack,
        "cases": cases,
        "all_security_immutable": all(case["security_immutable"] for case in cases),
        "max_invariant_error": max(case["security_invariant_error"] for case in cases),
        "all_finite": all(case["finite"] for case in cases),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attack", type=float, default=10.0)
    p.add_argument("--modes", nargs="+", default=["state_init", "security_only", "soft_only"])
    args = p.parse_args()
    result = run(args.seed, args.attack, tuple(args.modes))
    print(json.dumps(result, indent=2))
    if not result["all_security_immutable"] or not result["all_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
