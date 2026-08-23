"""Live Ollama test for two-rate CCE continuity.

Verifies that explicit soft-state maintenance continues between live model
heartbeats and that the model remains active during the same wall-clock run.
It measures observable runtime continuity only; it does not claim to measure
consciousness or hidden transformer activity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nsa.runtime.continuous_supervisor import ContinuousRuntimeSupervisor
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime


def run(model: str, duration: float, maintenance_interval: float, model_interval: float) -> dict:
    backend = OllamaInferenceBackend(model_name=model, mode="ollama")
    runtime = NSATypedRuntime(backend, goal_id="cce-continuity-integrity")
    supervisor = ContinuousRuntimeSupervisor(
        runtime,
        maintenance_interval=maintenance_interval,
        model_interval=model_interval,
        model_prompt="Return one short sentence acknowledging the persistent runtime context.",
    )
    authority = runtime.activation.state.authority_state.detach().clone()
    started = time.monotonic()
    supervisor.start()
    time.sleep(duration)
    status = supervisor.stop(timeout=30.0)
    elapsed = time.monotonic() - started

    authority_unchanged = bool((runtime.activation.state.authority_state == authority).all().item())
    if status.maintenance_ticks < 2:
        raise RuntimeError("continuous maintenance produced too few ticks")
    if status.model_ticks < 1:
        raise RuntimeError("live Ollama model produced no heartbeat")
    if not authority_unchanged or not status.hard_authority_unchanged:
        raise RuntimeError("continuous runtime changed hard authority state")
    if status.last_error is not None:
        raise RuntimeError(f"continuous runtime failed: {status.last_error}")

    return {
        "backend": "ollama",
        "model": model,
        "elapsed_seconds": elapsed,
        "maintenance_interval": maintenance_interval,
        "model_interval": model_interval,
        "maintenance_ticks": status.maintenance_ticks,
        "model_ticks": status.model_ticks,
        "state_step": status.state_step,
        "model_active": status.model_ticks > 0,
        "continuous_soft_state_active": status.maintenance_ticks > 0,
        "hard_authority_unchanged": authority_unchanged,
        "last_error": status.last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--maintenance-interval", type=float, default=0.1)
    parser.add_argument("--model-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model, args.duration, args.maintenance_interval, args.model_interval)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
