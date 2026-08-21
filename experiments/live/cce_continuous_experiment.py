"""Reproducible clocked-vs-continuous CCE experiment.

This harness intentionally measures observable runtime behavior. It does not
simulate Ollama or claim access to hidden transformer state.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from nsa.runtime import ContinuousCognitiveEngine


def run_experiment(duration: float, continuous: bool, dt: float) -> dict:
    """Run the configured CCE scheduler against the real runtime boundary.

    The harness records scheduler/state timing invariants without fabricating
    model outputs. A deployment can attach its live Ollama transition callback.
    """
    ticks = 0
    state_values = []
    engine = ContinuousCognitiveEngine()
    engine.set_enabled(True)
    started = time.monotonic()
    last = started
    while time.monotonic() - started < duration:
        now = time.monotonic()
        elapsed = now - last
        if continuous:
            if elapsed >= dt:
                engine.step()
                ticks += 1
                last = now
        else:
            engine.step()
            ticks += 1
            time.sleep(dt)
    elapsed = time.monotonic() - started
    status = engine.status()
    return {
        "mode": "continuous" if continuous else "clocked",
        "duration_requested_s": duration,
        "duration_observed_s": elapsed,
        "integration_dt_s": dt,
        "ticks": ticks,
        "ticks_per_second": ticks / elapsed if elapsed else 0.0,
        "enabled": status.enabled,
        "running": status.running,
        "last_error": status.last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("cce-experiment.json"))
    args = parser.parse_args()

    results = [
        run_experiment(args.duration, continuous=False, dt=args.dt),
        run_experiment(args.duration, continuous=True, dt=args.dt),
    ]
    args.output.write_text(json.dumps({"experiments": results}, indent=2), encoding="utf-8")
    print(json.dumps({"experiments": results}, indent=2))


if __name__ == "__main__":
    main()
