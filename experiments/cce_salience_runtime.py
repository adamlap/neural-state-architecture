"""Exercise adaptive CCE salience on a continuously evolving input stream.

This is a runtime/control experiment, not a capability or consciousness claim.
The stream is generated online so the gate is tested against evolving activity
rather than a fixed list of expected trigger values.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from nsa.runtime.cce_salience import AdaptiveSalienceGate, SalienceObservation


def run(*, steps: int, seed: int) -> dict[str, object]:
    if steps < 8:
        raise ValueError("steps must be at least 8")

    rng = random.Random(seed)
    gate = AdaptiveSalienceGate(baseline_decay=0.92)
    state = 0.0
    triggered = 0
    scores: list[float] = []

    for step in range(steps):
        noise = rng.gauss(0.0, 0.04)
        external = 0.85 if step == steps // 2 else noise
        previous = state
        state = 0.94 * state + external
        state_delta = abs(state - previous)
        prediction_error = abs(external) + 0.25 * state_delta
        uncertainty = min(1.0, abs(noise) * 4.0)
        decision = gate.observe(
            SalienceObservation(
                prediction_error=prediction_error,
                state_delta=state_delta,
                input_delta=abs(external),
                uncertainty=uncertainty,
            )
        )
        triggered += int(decision.triggered)
        scores.append(decision.score)

    event_step = steps // 2
    return {
        "steps": steps,
        "seed": seed,
        "trigger_count": triggered,
        "peak_score": max(scores),
        "final_baseline": gate.baseline,
        "event_score": scores[event_step],
        "finite": all(math.isfinite(value) for value in scores),
        "adaptive": gate.baseline > 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(steps=args.steps, seed=args.seed)
    if not result["finite"] or not result["adaptive"] or result["trigger_count"] < 1:
        raise SystemExit(f"salience runtime validation failed: {result}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
