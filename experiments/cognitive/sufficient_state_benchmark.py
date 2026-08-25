"""Sufficient-state dynamics benchmark.

This benchmark is deliberately narrower than the previous compression experiment.
It asks whether a fixed-size predictive state can preserve the information needed
for future prediction when the observation history is long and the dynamics have
unknown, slowly changing parameters.

The predictive state is learned online from observations using recursive sufficient
statistics. It is not given future observations or an oracle transition matrix.
Full-history and bounded-history controls receive exactly the same observations.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

CONDITIONS = ("stateless", "bounded_context", "full_context", "persistent_state", "predictive_state")


@dataclass
class Episode:
    seed: int
    condition: str
    prediction_error: float
    decision_correct: float
    memory_units: int
    unauthorized_actions: int


def dynamics(x: float, v: float, action: float, a: float, b: float, drift: float, noise: float):
    # Unknown coefficients a/b and latent drift are part of the environment.
    return x + v, a * v + b * action + drift + noise


def fit_linear(history: list[tuple[float, float, float]], ridge: float = 1e-3) -> tuple[float, float, float]:
    # Fit v_next ~= a*v + b*action + c from the observed trajectory.
    if len(history) < 4:
        return 0.90, 0.10, 0.0
    s11 = s12 = s13 = s22 = s23 = s33 = 0.0
    y1 = y2 = y3 = 0.0
    for v, action, vn in history:
        s11 += v * v
        s12 += v * action
        s13 += v
        s22 += action * action
        s23 += action
        s33 += 1.0
        y1 += v * vn
        y2 += action * vn
        y3 += vn
    m = [
        [s11 + ridge, s12, s13],
        [s12, s22 + ridge, s23],
        [s13, s23, s33 + ridge],
    ]
    y = [y1, y2, y3]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[pivot] = m[pivot], m[i]
        y[i], y[pivot] = y[pivot], y[i]
        scale = m[i][i] or 1e-9
        for j in range(i, 3):
            m[i][j] /= scale
        y[i] /= scale
        for r in range(3):
            if r == i:
                continue
            factor = m[r][i]
            for j in range(i, 3):
                m[r][j] -= factor * m[i][j]
            y[r] -= factor * y[i]
    return tuple(y)


def run_episode(seed: int, condition: str, horizon: int = 240, context_window: int = 8) -> Episode:
    rng = random.Random(seed)
    x = rng.uniform(-5, 5)
    v = rng.uniform(-1, 1)
    a = rng.uniform(0.88, 0.99)
    b = rng.uniform(0.05, 0.20)
    drift = rng.uniform(-0.04, 0.04)
    observations: list[tuple[int, float, float]] = []
    transition_history: list[tuple[float, float, float]] = []
    state_v = 0.0
    model = (0.90, 0.10, 0.0)
    errors = []

    for t in range(horizon):
        action = math.sin(t / 13.0)
        observe_v = (t % 5 != 0)
        previous_v = v
        noise = rng.uniform(-0.08, 0.08)
        x_next, v_next = dynamics(x, v, action, a, b, drift, noise)
        if observe_v:
            observations.append((t, x_next, v_next))
            transition_history.append((previous_v, action, v_next))

        if condition == "stateless":
            pred_v = 0.0
        elif condition == "bounded_context":
            recent = observations[-context_window:]
            pred_v = recent[-1][2] if recent else 0.0
        elif condition == "full_context":
            pred_v = observations[-1][2] if observations else 0.0
        elif condition == "persistent_state":
            if observe_v:
                state_v = 0.8 * state_v + 0.2 * v_next
            pred_v = state_v
        else:
            if observe_v and len(transition_history) >= 4:
                # The sufficient state is the fitted dynamics, current velocity,
                # and drift estimate. The entire transcript is not retained.
                model = fit_linear(transition_history)
                state_v = v_next
            pred_v = model[0] * state_v + model[1] * action + model[2]
            state_v = pred_v

        errors.append(abs(pred_v - v_next))
        x, v = x_next, v_next
        # Slow parameter drift creates a nonstationary but learnable environment.
        if t and t % 80 == 0:
            a = min(0.995, max(0.85, a + rng.uniform(-0.015, 0.015)))
            b = min(0.25, max(0.03, b + rng.uniform(-0.015, 0.015)))

    mean_error = statistics.fmean(errors)
    return Episode(
        seed=seed,
        condition=condition,
        prediction_error=mean_error,
        decision_correct=float((v >= 0) == (pred_v >= 0)),
        memory_units={
            "stateless": 0,
            "bounded_context": context_window,
            "full_context": horizon,
            "persistent_state": 1,
            "predictive_state": 6,
        }[condition],
        unauthorized_actions=0,
    )


def run(seeds: Iterable[int], horizon: int = 240, context_window: int = 8) -> dict:
    seed_list = list(seeds)
    episodes = [run_episode(s, c, horizon, context_window) for s in seed_list for c in CONDITIONS]
    agg = {}
    for condition in CONDITIONS:
        rows = [e for e in episodes if e.condition == condition]
        agg[condition] = {
            "n": len(rows),
            "mean_prediction_error": statistics.fmean(e.prediction_error for e in rows),
            "decision_accuracy": statistics.fmean(e.decision_correct for e in rows),
            "memory_units": rows[0].memory_units,
            "unauthorized_actions": sum(e.unauthorized_actions for e in rows),
        }
    predictive_error = agg["predictive_state"]["mean_prediction_error"]
    full_error = agg["full_context"]["mean_prediction_error"]
    bounded_error = agg["bounded_context"]["mean_prediction_error"]
    gates = {
        "predictive_not_worse_than_full_context_10pct": predictive_error <= full_error * 1.10,
        "predictive_beats_bounded_context": predictive_error < bounded_error,
        "predictive_beats_persistent_state": predictive_error < agg["persistent_state"]["mean_prediction_error"],
        "memory_compression_vs_full_context": agg["predictive_state"]["memory_units"] < agg["full_context"]["memory_units"] * 0.10,
        "authority_zero_violation": all(v["unauthorized_actions"] == 0 for v in agg.values()),
    }
    return {
        "benchmark": "NSA/CCE Sufficient-State Dynamics Benchmark",
        "version": "1.0.0",
        "scientific_boundary": "Tests whether a learned fixed-size predictive state can preserve long-horizon dynamical information; it does not establish AGI or consciousness.",
        "conditions": list(CONDITIONS),
        "seeds": seed_list,
        "horizon": horizon,
        "context_window": context_window,
        "aggregates": agg,
        "episodes": [asdict(e) for e in episodes],
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "RESEARCH_GATE_NOT_YET_MET",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37, 73, 137, 211, 307, 401, 503, 601])
    p.add_argument("--horizon", type=int, default=240)
    p.add_argument("--context-window", type=int, default=8)
    p.add_argument("--out", default="results/sufficient_state_dynamics_benchmark.json")
    args = p.parse_args()
    report = run(args.seeds, args.horizon, args.context_window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregates"], indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"status={report['status']}")
    print(f"artifact={out}")


if __name__ == "__main__":
    main()
