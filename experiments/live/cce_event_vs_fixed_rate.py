"""Compare event-driven and fixed-rate Ollama invocation on one live CCE stream.

This is an empirical runtime experiment, not a cognition or consciousness claim.
Both conditions evolve a real ContinuousStateField using measured wall-clock time.
The same seeded external event stream is replayed into each condition. The fixed
condition invokes Ollama every observation; the event-driven condition invokes it
only when AdaptiveSalienceGate fires.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from nsa.runtime.cce_loop import ClosedLoopCognitiveInvoker
from nsa.runtime.cce_salience import SalienceObservation
from nsa.runtime.continuous_state_field import ContinuousStateField
from nsa.runtime.inference.ollama import OllamaInferenceBackend


def _field(state: torch.Tensor, external: float | None) -> torch.Tensor:
    """Transparent control dynamics: damped state plus asynchronous input."""
    forcing = 0.0 if external is None else float(external)
    return -0.25 * state + torch.full_like(state, forcing)


def _run_condition(
    *, model: str, events: list[float], interval: float, event_driven: bool, max_tokens: int
) -> dict:
    backend = OllamaInferenceBackend(model_name=model, timeout_sec=60.0)
    invoker = ClosedLoopCognitiveInvoker(backend)
    field = ContinuousStateField(
        torch.zeros(4),
        _field,
        integration_cadence_seconds=min(0.02, max(interval / 5.0, 0.005)),
        enabled=True,
    )
    field.start()
    responses = 0
    latencies: list[float] = []
    observations = 0
    try:
        # Let the real field establish its first wall-clock timestamp.
        time.sleep(max(interval * 0.5, 0.02))
        for index, forcing in enumerate(events):
            field.inject(forcing)
            time.sleep(interval)
            state = field.state
            magnitude = float(torch.linalg.vector_norm(state).item())
            previous = 0.0 if index == 0 else abs(events[index - 1])
            input_delta = abs(forcing - previous)
            observation = SalienceObservation(
                prediction_error=input_delta + magnitude * 0.1,
                state_delta=magnitude,
                input_delta=input_delta,
                uncertainty=min(1.0, 0.1 + input_delta),
            )
            prompt = (
                "You are observing a continuously evolving cognitive substrate. "
                f"step={index}; state_norm={magnitude:.6f}; input_delta={input_delta:.6f}. "
                "Return one concise observation of the current state."
            )
            if event_driven:
                start = time.perf_counter()
                result = invoker.observe(observation, prompt, max_tokens=max_tokens, temperature=0.0)
                elapsed = (time.perf_counter() - start) * 1000.0
                if result.triggered:
                    responses += 1
                    latencies.append(elapsed)
            else:
                start = time.perf_counter()
                output = backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
                elapsed = (time.perf_counter() - start) * 1000.0
                if output.text.strip():
                    responses += 1
                latencies.append(elapsed)
            observations += 1
    finally:
        field.stop(timeout=5.0)

    status = field.status()
    return {
        "condition": "event_driven" if event_driven else "fixed_rate",
        "observations": observations,
        "invocations": responses,
        "invocation_rate": responses / observations if observations else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "field_integrations": status.integration_count,
        "elapsed_seconds": status.elapsed_seconds,
        "running_after_stop": status.running,
        "last_error": status.last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.interval <= 0:
        raise ValueError("steps must be >= 2 and interval must be > 0")

    rng = random.Random(args.seed)
    events = [rng.uniform(-1.0, 1.0) if i % 3 == 0 else rng.uniform(-0.15, 0.15) for i in range(args.steps)]
    fixed = _run_condition(
        model=args.model, events=events, interval=args.interval, event_driven=False, max_tokens=args.max_tokens
    )
    event_driven = _run_condition(
        model=args.model, events=events, interval=args.interval, event_driven=True, max_tokens=args.max_tokens
    )

    result = {
        "backend_mode": "ollama",
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "interval_seconds": args.interval,
        "same_event_stream": True,
        "fixed_rate": fixed,
        "event_driven": event_driven,
        "invocation_reduction": fixed["invocations"] - event_driven["invocations"],
        "invocation_reduction_fraction": (
            (fixed["invocations"] - event_driven["invocations"]) / fixed["invocations"]
            if fixed["invocations"]
            else 0.0
        ),
        "both_clean_shutdown": fixed["running_after_stop"] is False and event_driven["running_after_stop"] is False,
        "both_error_free": fixed["last_error"] is None and event_driven["last_error"] is None,
    }
    if not result["both_clean_shutdown"] or not result["both_error_free"]:
        raise RuntimeError("continuous field did not terminate cleanly")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
