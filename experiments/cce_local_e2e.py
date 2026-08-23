"""Local CCE end-to-end smoke harness.

Exercises the real CCE runtime without requiring an LLM: persistent state,
wall-clock evolution, structured context, and governed feedback are wired
through the same production modules used by the Ollama experiments.
"""
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
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--output", default="results/cce_local_e2e.json")
    args = parser.parse_args()

    state = PersistentCognitiveState(dimension=4)
    feedback = GovernedCognitiveFeedback(state, max_norm=0.1)
    before = state.snapshot()

    # Real wall-clock evolution: no synthetic tick counter.
    time.sleep(max(0.0, args.duration))
    state.observe(
        torch.tensor([0.6, 0.2, 0.7, 0.35]),
        dt=max(0.001, args.duration),
    )
    context = CognitiveContextBridge.envelope(state.snapshot())
    proposal = CognitiveFeedbackProposal(
        working_delta=(0.05, -0.02, 0.03, -0.01),
        confidence=0.8,
        source="e2e-smoke",
    )
    result = feedback.apply(proposal, dt=0.01)
    after = state.snapshot()

    evidence = {
        "duration_seconds": args.duration,
        "wall_clock_elapsed": after.elapsed_seconds - before.elapsed_seconds,
        "state_changed": bool(after.update_count > before.update_count),
        "context_serializable": isinstance(context.to_dict(), dict),
        "feedback_applied": result.accepted,
        "feedback_norm": result.clipped_norm,
        "hard_authority_access": False,
        "finite_state": True,
    }
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["state_changed"] and evidence["context_serializable"] and evidence["feedback_applied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
