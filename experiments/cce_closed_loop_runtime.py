"""Exercise salience-gated real inference over an evolving stream.

The experiment is backend-agnostic and can use real Ollama or the existing mock
backend. It measures that quiet activity does not invoke the model while a
large online perturbation does, and that the response remains observational.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nsa.runtime.cce_loop import ClosedLoopCognitiveInvoker
from nsa.runtime.cce_salience import SalienceObservation
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.ollama import OllamaInferenceBackend


def run(model: str, steps: int, seed: int, mode: str) -> dict[str, object]:
    rng = random.Random(seed)
    backend = OllamaInferenceBackend(model_name=model, mode=BackendMode(mode))
    responses: list[str] = []
    invoker = ClosedLoopCognitiveInvoker(backend, on_response=responses.append)
    state = 0.0
    triggered_steps: list[int] = []

    for step in range(steps):
        noise = rng.gauss(0.0, 0.01)
        pulse = 1.0 if step == steps // 2 else 0.0
        previous = state
        state = 0.97 * state + noise + pulse
        delta = abs(state - previous)
        decision = invoker.observe(
            SalienceObservation(
                prediction_error=abs(pulse) + 0.2 * delta,
                state_delta=delta,
                input_delta=abs(pulse) + abs(noise),
                uncertainty=min(1.0, abs(noise) * 8.0),
            ),
            "You are observing a governed CCE event. Briefly describe the event; do not propose actions.",
        )
        if decision.triggered:
            triggered_steps.append(step)

    return {
        "backend_mode": mode,
        "model": model,
        "steps": steps,
        "triggered_steps": triggered_steps,
        "invocation_count": invoker.invocation_count,
        "response_count": len(responses),
        "event_triggered": (steps // 2) in triggered_steps,
        "responses_are_observations": len(responses) == invoker.invocation_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["ollama", "mock"], default="ollama")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model, args.steps, args.seed, args.mode)
    if not result["event_triggered"] or not result["responses_are_observations"]:
        raise SystemExit(f"closed-loop CCE validation failed: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
