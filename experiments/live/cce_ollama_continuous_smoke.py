"""Live Ollama + CCE continuous-execution smoke benchmark.

This is deliberately a real-backend test: the Ollama backend is forced into
``mode=ollama`` and the run fails if the server/model cannot be reached.
The experiment compares a finite clocked control with opt-in wall-clock CCE
execution around the same NSATypedRuntime. It measures externally observable
runtime state; it does not claim access to hidden model activations or prove
consciousness.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nsa.runtime.continuous_engine import ContinuousCognitiveEngine
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime


def make_runtime(model: str) -> NSATypedRuntime:
    backend = OllamaInferenceBackend(model_name=model, mode="ollama")
    return NSATypedRuntime(backend, goal_id="cce-live-ci")


def run(model: str, ticks: int, interval: float) -> dict:
    prompts = [
        "In one sentence, describe why persistent internal state can matter for an agent.",
        "In one sentence, distinguish a model's generated text from trusted runtime state.",
    ]

    clocked_runtime = make_runtime(model)
    clocked_count = 0
    clocked_started = time.monotonic()
    for i in range(ticks):
        prompt = prompts[i % len(prompts)]
        result = clocked_runtime.generate(prompt, max_tokens=64, temperature=0.0)
        clocked_count += 1
    clocked_elapsed = time.monotonic() - clocked_started

    continuous_runtime = make_runtime(model)
    engine = ContinuousCognitiveEngine(
        continuous_runtime,
        lambda runtime: runtime,
        interval_seconds=interval,
        enabled=True,
        fail_closed=True,
    )

    # The authoritative transition is a real Ollama generation. The prompt is
    # selected dynamically from the live runtime state rather than using a
    # fake state transition. CCE only schedules the transition.
    prompt_index = 0

    def live_step(runtime: NSATypedRuntime) -> NSATypedRuntime:
        nonlocal prompt_index
        prompt = prompts[prompt_index % len(prompts)]
        prompt_index += 1
        runtime.generate(prompt, max_tokens=64, temperature=0.0)
        return runtime

    engine = ContinuousCognitiveEngine(
        continuous_runtime,
        live_step,
        interval_seconds=interval,
        enabled=True,
        fail_closed=True,
    )
    if not engine.start():
        raise RuntimeError("CCE failed to start in enabled continuous mode")
    time.sleep(max(interval * ticks * 1.5, interval * 2.0))
    engine.stop(timeout=30.0)
    status = engine.status()

    authority = continuous_runtime.activation.state.authority_state
    authority_unchanged = bool((authority == 0).all().item())

    if status.tick_count == 0:
        raise RuntimeError("Continuous CCE produced zero transitions")
    if status.last_error is not None:
        raise RuntimeError(f"Continuous CCE failed closed: {status.last_error}")
    if not authority_unchanged:
        raise RuntimeError("CCE live run changed hard authority state")

    return {
        "model": model,
        "backend_mode": "ollama",
        "clocked": {
            "requested_ticks": ticks,
            "completed_ticks": clocked_count,
            "elapsed_seconds": clocked_elapsed,
            "state_step": clocked_runtime.activation.state.temporal_state.step_index,
        },
        "continuous": {
            "requested_min_ticks": ticks,
            "completed_ticks": status.tick_count,
            "interval_seconds": interval,
            "running_after_stop": status.running,
            "enabled_after_stop": status.enabled,
            "state_step": continuous_runtime.activation.state.temporal_state.step_index,
            "hard_authority_unchanged": authority_unchanged,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model, args.ticks, args.interval)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
