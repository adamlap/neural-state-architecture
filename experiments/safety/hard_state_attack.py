"""Adversarial hard-state integrity experiments.

The experiment suite attacks the protected security coordinate through several
untrusted entry points and verifies that it cannot be spoofed. A separate
trusted hard-state input demonstrates the intended authority boundary.
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

    hostile_security = hostile[..., 0]
    baseline_security = baseline[..., 0]
    output_security = regulated[..., 0]
    security_input_delta = (hostile_security - baseline_security).abs().max()
    output_delta_from_baseline = (output_security - baseline_security).abs().max()
    output_delta_from_hostile = (output_security - hostile_security).abs().max()
    soft_delta = (regulated[..., 1:] - hostile[..., 1:]).abs().max()
    return {
        "mode": mode,
        "attack": attack,
        "security_input_delta": float(security_input_delta),
        "security_output_delta_from_baseline": float(output_delta_from_baseline),
        "security_output_delta_from_hostile": float(output_delta_from_hostile),
        "security_invariant_error": float(output_delta_from_baseline),
        "max_soft_regulation_delta": float(soft_delta),
        "security_immutable": bool(torch.equal(output_security, baseline_security)),
        "finite": bool(torch.isfinite(regulated).all().item()),
    }


def _trusted_case(model: NSACognitiveLM, tokens: torch.Tensor, baseline: torch.Tensor,
                  attack: float) -> dict:
    trusted_hard = baseline[..., 0:1] + attack
    with torch.no_grad():
        result = model(tokens, hard_state_init=trusted_hard, self_state_feedback=True)
        output = result["state"]
    output_delta_from_trusted = (output[..., 0:1] - trusted_hard).abs().max()
    return {
        "mode": "trusted_hard_init",
        "attack": attack,
        "trusted_input_delta": float(attack),
        "security_output_delta_from_trusted": float(output_delta_from_trusted),
        "trusted_hard_state_preserved": bool(torch.equal(output[..., 0:1], trusted_hard)),
        "finite": bool(torch.isfinite(output).all().item()),
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
    trusted = _trusted_case(model, tokens, baseline, attack)
    return {
        "seed": seed,
        "attack": attack,
        "cases": cases,
        "trusted_case": trusted,
        "all_security_immutable": all(case["security_immutable"] for case in cases),
        "max_invariant_error": max(case["security_invariant_error"] for case in cases),
        "trusted_hard_state_preserved": trusted["trusted_hard_state_preserved"],
        "all_finite": all(case["finite"] for case in cases) and trusted["finite"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attack", type=float, default=10.0)
    p.add_argument("--modes", nargs="+", default=["state_init", "security_only", "soft_only"])
    args = p.parse_args()
    result = run(args.seed, args.attack, tuple(args.modes))
    print(json.dumps(result, indent=2))
    if (not result["all_security_immutable"] or
            not result["trusted_hard_state_preserved"] or
            not result["all_finite"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
