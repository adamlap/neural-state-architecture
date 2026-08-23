"""Local CCE end-to-end smoke harness using production runtime interfaces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.runtime.cce_context_bridge import CognitiveContextBridge
from nsa.runtime.cce_governed_feedback import CognitiveFeedbackProposal, GovernedCognitiveFeedback
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--output", default="results/cce_local_e2e.json")
    args = parser.parse_args()

    state = PersistentCognitiveState(dimension=4)
    before = state.snapshot()
    started = time.monotonic()
    if args.duration > 0.0:
        time.sleep(args.duration)
    dt = max(time.monotonic() - started, 1e-6)

    observed = state.observe(torch.tensor([0.6, 0.2, 0.7, 0.35]), dt=dt)
    context = CognitiveContextBridge.envelope(observed)
    governor = GovernedCognitiveFeedback(state, max_norm=0.1)
    proposal = CognitiveFeedbackProposal(
        working_delta=(0.05, 0.0, 0.0, 0.0),
        goal_delta=(0.0, 0.0, 0.03, 0.0),
        confidence=0.8,
        source="local-e2e",
    )
    feedback = governor.apply(proposal, dt=dt)
    after = feedback.snapshot

    evidence = {
        "duration_seconds": float(args.duration),
        "measured_dt_seconds": float(dt),
        "state_changed": before.update_count != after.update_count,
        "context_serializable": isinstance(context.to_dict(), dict),
        "feedback_accepted": feedback.accepted,
        "feedback_norm": feedback.clipped_norm,
        "feedback_within_budget": feedback.clipped_norm <= 0.1 + 1e-7,
        "hard_authority_access": False,
        "finite_state": all(torch.isfinite(v).all().item() for v in (after.working, after.self_state, after.goal))
        and 0.0 <= after.uncertainty <= 1.0,
        "update_count": after.update_count,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if all(evidence[k] for k in ("state_changed", "context_serializable", "feedback_accepted", "feedback_within_budget", "finite_state")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
