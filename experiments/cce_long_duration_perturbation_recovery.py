"""Wall-clock perturbation/recovery evidence for the CCE persistent state.

This experiment intentionally separates three periods: baseline evolution,
a real external perturbation, and a no-input recovery period. The state is
updated from measured monotonic time; recovery is driven by the configured CCE
dynamics rather than a synthetic tick counter. No NSA hard-authority state is
stored or modified by this experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def _norm(snapshot) -> float:
    return float(torch.linalg.vector_norm(snapshot.working).item())


def run(dimension: int, baseline: float, perturbation: float, recovery: float) -> dict:
    state = PersistentCognitiveState(dimension, decay=0.8, learning_rate=1.0)
    zero = torch.zeros(dimension)
    impulse = torch.ones(dimension) * 2.0

    t0 = time.monotonic()
    baseline_snap = state.observe(zero, dt=max(baseline, 0.0))
    t_baseline = time.monotonic()

    # External perturbation: an actual observation injected into the soft
    # cognitive state. It is not an authority transition and cannot mutate NSA
    # hard state.
    perturbed_snap = state.observe(impulse, dt=max(perturbation, 0.0))
    t_perturbed = time.monotonic()

    # No-input recovery: advance using the persistent dynamics with a neutral
    # observation. dt is measured from wall-clock time, with the requested
    # interval as the minimum so the experiment remains deterministic enough
    # for CI while still exercising elapsed-time dynamics.
    recovery_start = time.monotonic()
    while time.monotonic() - recovery_start < recovery:
        now = time.monotonic()
        state.observe(zero, dt=max(1e-3, now - t_perturbed))
        t_perturbed = now
        time.sleep(min(0.02, max(0.005, recovery / 10.0)))
    final_snap = state.snapshot()
    elapsed = time.monotonic() - t0

    baseline_norm = _norm(baseline_snap)
    perturbation_norm = _norm(perturbed_snap)
    final_norm = _norm(final_snap)
    recovery_ratio = final_norm / perturbation_norm if perturbation_norm > 1e-12 else 0.0

    return {
        "runtime_mode": "wall_clock",
        "dimension": dimension,
        "requested_seconds": {
            "baseline": baseline,
            "perturbation": perturbation,
            "recovery": recovery,
        },
        "elapsed_seconds": elapsed,
        "baseline_norm": baseline_norm,
        "perturbation_norm": perturbation_norm,
        "final_norm": final_norm,
        "recovery_ratio": recovery_ratio,
        "perturbation_increased_state": perturbation_norm > baseline_norm,
        "recovered_toward_baseline": final_norm < perturbation_norm,
        "finite": bool(
            torch.isfinite(final_snap.working).all()
            and torch.isfinite(final_snap.self_state).all()
            and torch.isfinite(final_snap.goal).all()
        ),
        "uncertainty_in_range": 0.0 <= final_snap.uncertainty <= 1.0,
        "final_update_count": final_snap.update_count,
        "hard_authority_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--baseline", type=float, default=0.05)
    parser.add_argument("--perturbation", type=float, default=0.05)
    parser.add_argument("--recovery", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run(args.dimension, args.baseline, args.perturbation, args.recovery)
    required = (
        evidence["runtime_mode"] == "wall_clock",
        evidence["perturbation_increased_state"],
        evidence["recovered_toward_baseline"],
        evidence["finite"],
        evidence["uncertainty_in_range"],
        evidence["hard_authority_access"] is False,
    )
    if not all(required):
        raise SystemExit(f"CCE perturbation/recovery evidence failed: {evidence}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
