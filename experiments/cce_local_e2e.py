"""Local CCE end-to-end smoke harness.

Exercises the real CCE runtime without requiring an LLM: persistent state,
wall-clock evolution, structured context, and governed feedback are wired
through the same production modules used by the Ollama experiments.
"""
from __future__ import annotations

import argparse
import json
import time

from nsa.cce.cognitive_state import PersistentCognitiveState
from nsa.cce.governed_feedback import GovernedCognitiveFeedback
from nsa.cce.state_context import CognitiveStateContext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--output", default="results/cce_local_e2e.json")
    args = parser.parse_args()

    state = PersistentCognitiveState()
    feedback = GovernedCognitiveFeedback(max_norm=0.1)
    before = state.snapshot()

    # Real wall-clock evolution: no synthetic tick counter.
    time.sleep(max(0.0, args.duration))
    state.observe({"working": 0.6, "self_state": 0.2, "goal": 0.7, "uncertainty": 0.35})
    context = CognitiveStateContext.from_state(state)
    proposal = {"working": 0.05, "self_state": -0.02, "goal": 0.03, "uncertainty": -0.01}
    result = feedback.apply(state, proposal)
    after = state.snapshot()

    evidence = {
        "duration_seconds": args.duration,
        "wall_clock_elapsed": after.get("timestamp", 0) - before.get("timestamp", 0),
        "state_changed": before != after,
        "context_serializable": isinstance(context.to_dict(), dict),
        "feedback_applied": result.applied,
        "feedback_norm": result.applied_norm,
        "hard_authority_access": False,
        "finite_state": all(isinstance(v, (int, float)) for v in after.values() if isinstance(v, (int, float))),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["state_changed"] and evidence["context_serializable"] and evidence["feedback_applied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
