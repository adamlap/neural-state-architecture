"""Runtime evidence for bounded CCE feedback without hard-state access."""
from __future__ import annotations

import argparse
import json
import time

import torch

from nsa.runtime.cce_governed_feedback import CognitiveFeedbackProposal, GovernedCognitiveFeedback
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def run(dimension: int, duration: float, max_norm: float) -> dict:
    state = PersistentCognitiveState(dimension, learning_rate=1.0)
    gate = GovernedCognitiveFeedback(state, max_norm=max_norm)
    start = time.monotonic()
    applications = 0
    max_observed_norm = 0.0
    while time.monotonic() - start < duration:
        proposal = CognitiveFeedbackProposal(
            working_delta=tuple(0.05 for _ in range(dimension)),
            goal_delta=tuple(0.02 for _ in range(dimension)),
            confidence=0.5,
            source="runtime-experiment",
        )
        result = gate.apply(proposal, dt=0.01)
        applications += 1
        max_observed_norm = max(max_observed_norm, result.clipped_norm)
        time.sleep(0.005)
    snapshot = state.snapshot()
    return {
        "dimension": dimension,
        "duration_seconds": time.monotonic() - start,
        "applications": applications,
        "state_updates": snapshot.update_count,
        "state_changed": bool(torch.linalg.vector_norm(snapshot.working).item() > 0.0),
        "max_feedback_norm": max_observed_norm,
        "max_norm_budget": max_norm,
        "finite": bool(torch.isfinite(snapshot.working).all() and torch.isfinite(snapshot.goal).all()),
        "uncertainty_in_range": 0.0 <= snapshot.uncertainty <= 1.0,
        "hard_authority_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--duration", type=float, default=0.2)
    parser.add_argument("--max-norm", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = run(args.dimension, args.duration, args.max_norm)
    if not evidence["state_changed"] or not evidence["finite"] or not evidence["uncertainty_in_range"]:
        raise SystemExit("governed feedback evidence failed")
    if evidence["max_feedback_norm"] > args.max_norm + 1e-6:
        raise SystemExit("feedback budget violated")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
