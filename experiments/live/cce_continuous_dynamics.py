"""Empirical experiment for genuine input-free continuous CCE dynamics.

This is intentionally independent of Ollama. The question is whether the
continuous state substrate evolves between external events.  A small injected
ODE is used only as a transparent control field; it is not presented as a
model of cognition.

The experiment records the state trajectory, measured elapsed time, and an
asynchronous perturbation.  It exits non-zero if the state fails to evolve
without input or if the perturbation is not reflected in the trajectory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.runtime.continuous_state_field import ContinuousStateField


def run(duration: float, cadence: float, perturb_at: float) -> dict[str, object]:
    # Transparent stable control dynamics: dx/dt = -0.2x + u.
    def field(state: torch.Tensor, external: float | None) -> torch.Tensor:
        u = 0.0 if external is None else float(external)
        return -0.2 * state + u

    state = torch.tensor([1.0], dtype=torch.float64)
    engine = ContinuousStateField(
        state,
        field,
        integration_cadence_seconds=cadence,
        enabled=True,
    )
    engine.start()

    trajectory = [(0.0, float(engine.state.item()))]
    start = time.monotonic()
    injected = False
    while time.monotonic() - start < duration:
        elapsed = time.monotonic() - start
        if not injected and elapsed >= perturb_at:
            engine.inject(0.75)
            injected = True
        time.sleep(min(cadence, 0.01))
        trajectory.append((elapsed, float(engine.state.item())))

    engine.stop(timeout=2.0)
    final = engine.state
    status = engine.status()

    pre = [value for t, value in trajectory if t < min(perturb_at, duration)]
    no_input_evolved = len(pre) >= 2 and abs(pre[-1] - pre[0]) > 1e-6
    perturbation_observed = injected and abs(float(final.item()) - pre[-1]) > 1e-6

    result = {
        "experiment": "cce_continuous_dynamics",
        "dynamics": "dx/dt = -0.2*x + u",
        "duration_seconds": duration,
        "cadence_seconds": cadence,
        "perturb_at_seconds": perturb_at,
        "integration_count": status.integration_count,
        "measured_elapsed_seconds": status.elapsed_seconds,
        "no_input_evolved": no_input_evolved,
        "perturbation_injected": injected,
        "perturbation_observed": perturbation_observed,
        "running_after_stop": status.running,
        "enabled_after_stop": status.enabled,
        "last_error": status.last_error,
        "initial_state": 1.0,
        "final_state": float(final.item()),
        "trajectory": trajectory,
    }
    if not no_input_evolved:
        raise RuntimeError("continuous field did not evolve during input-free interval")
    if not perturbation_observed:
        raise RuntimeError("asynchronous perturbation was not reflected in state")
    if status.running:
        raise RuntimeError("continuous field remained running after stop")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--cadence", type=float, default=0.02)
    parser.add_argument("--perturb-at", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.duration, args.cadence, args.perturb_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}, indent=2))


if __name__ == "__main__":
    main()
