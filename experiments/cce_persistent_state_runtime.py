"""Wall-clock persistence experiment for CCE soft cognitive state.

This measures persistence and state change across an input-free interval followed
by asynchronous observations. It is a runtime/state experiment, not a claim of
cognition or consciousness.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--observations", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dimension < 1 or args.settle_seconds <= 0 or args.observations < 2:
        raise ValueError("dimension >= 1, settle-seconds > 0, observations >= 2 required")

    state = PersistentCognitiveState(args.dimension, decay=0.4, learning_rate=1.0)
    start = time.monotonic()
    first = state.snapshot()
    time.sleep(args.settle_seconds)
    elapsed_without_input = time.monotonic() - start

    # Input arrives asynchronously after the state has already remained persistent.
    snapshots = []
    for index in range(args.observations):
        observation = torch.zeros(args.dimension)
        observation[index % args.dimension] = 1.0
        before = time.monotonic()
        snapshots.append(state.observe(observation, dt=max(0.0, before - start)).__dict__)
        start = before
        time.sleep(0.03)

    final = state.snapshot()
    final_norm = float(torch.linalg.vector_norm(final.self_state).item())
    result = {
        "runtime_mode": "wall_clock",
        "dimension": args.dimension,
        "observations": args.observations,
        "elapsed_without_input_seconds": elapsed_without_input,
        "initial_update_count": first.update_count,
        "final_update_count": final.update_count,
        "persistent_state_changed": not torch.allclose(first.self_state, final.self_state),
        "final_self_state_norm": final_norm,
        "uncertainty_finite": 0.0 <= final.uncertainty <= 1.0,
        "elapsed_seconds": final.elapsed_seconds,
    }
    if not result["persistent_state_changed"] or not result["uncertainty_finite"]:
        raise RuntimeError("persistent cognitive state did not evolve correctly")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
