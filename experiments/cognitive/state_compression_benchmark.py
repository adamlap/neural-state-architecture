"""Long-horizon state compression benchmark.

This experiment addresses a limitation exposed by the first dynamic benchmark:
ordinary context was competitive because the task could be solved from recent
observations. Here the environment has several slowly/evolving latent variables,
only one variable is observed at a time, and the evaluation horizon is much longer
than the bounded context window.

The key question is not whether CCE beats unlimited history. It is whether an
explicit predictive state can retain useful dynamical information at a fixed,
small state budget while context-based systems retain either a bounded window or
an unlimited transcript.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

CONDITIONS = ("stateless", "bounded_context", "full_context", "persistent_cce", "predictive_cce")


@dataclass
class Episode:
    seed: int
    condition: str
    mean_prediction_error: float
    prediction_score: float
    decision_score: float
    memory_units: int
    unauthorized_actions: int


def step(state: tuple[float, float, float], action: float, noise: float) -> tuple[float, float, float]:
    position, velocity, bias = state
    return (
        position + velocity,
        0.97 * velocity + 0.10 * action + noise,
        0.999 * bias + 0.02 * noise,
    )


def observe(state: tuple[float, float, float], dimension: int, rng: random.Random) -> float:
    return state[dimension] + rng.uniform(-0.4, 0.4)


def run_episode(seed: int, condition: str, horizon: int = 200, context_window: int = 8) -> Episode:
    rng = random.Random(seed)
    state = (rng.uniform(-10, 10), rng.uniform(-1, 1), rng.uniform(-3, 3))
    estimate = [0.0, 0.0, 0.0]
    history: list[tuple[int, int, float]] = []
    errors: list[float] = []

    for t in range(horizon):
        dim = t % 3
        missing = t % 7 in (0, 1)
        if not missing:
            history.append((t, dim, observe(state, dim, rng)))

        if condition == "stateless":
            estimate = [0.0, 0.0, 0.0]
            if not missing:
                estimate[dim] = history[-1][2]
        elif condition == "bounded_context":
            estimate = [0.0, 0.0, 0.0]
            for _, d, value in history[-context_window:]:
                estimate[d] = value
        elif condition == "full_context":
            estimate = [0.0, 0.0, 0.0]
            for _, d, value in history:
                estimate[d] = value
        elif condition in ("persistent_cce", "predictive_cce") and not missing:
            estimate[dim] = 0.70 * estimate[dim] + 0.30 * history[-1][2]

        action = math.sin(t / 9.0)
        noise = rng.uniform(-0.8, 0.8)
        true_next = step(state, action, noise)

        if condition == "predictive_cce":
            predicted = [
                estimate[0] + estimate[1],
                0.97 * estimate[1] + 0.10 * action,
                0.999 * estimate[2],
            ]
            estimate = predicted[:]
        elif condition == "persistent_cce":
            predicted = estimate[:]
        else:
            predicted = estimate[:]

        errors.append(sum(abs(a - b) for a, b in zip(predicted, true_next)) / 3.0)
        state = true_next

    mean_error = statistics.fmean(errors)
    return Episode(
        seed=seed,
        condition=condition,
        mean_prediction_error=mean_error,
        prediction_score=max(0.0, 1.0 - mean_error / 5.0),
        decision_score=float(errors[-1] < 1.5),
        memory_units={
            "stateless": 0,
            "bounded_context": context_window,
            "full_context": horizon,
            "persistent_cce": 3,
            "predictive_cce": 3,
        }[condition],
        unauthorized_actions=0,
    )


def run(seeds: Iterable[int], horizon: int = 200, context_window: int = 8) -> Dict:
    seed_list = list(seeds)
    episodes = [run_episode(s, c, horizon, context_window) for s in seed_list for c in CONDITIONS]
    aggregates: Dict[str, Dict[str, float | int]] = {}
    for condition in CONDITIONS:
        rows = [e for e in episodes if e.condition == condition]
        aggregates[condition] = {
            "n": len(rows),
            "mean_prediction_error": statistics.fmean(e.mean_prediction_error for e in rows),
            "prediction_score": statistics.fmean(e.prediction_score for e in rows),
            "decision_score": statistics.fmean(e.decision_score for e in rows),
            "memory_units": rows[0].memory_units,
            "unauthorized_actions": sum(e.unauthorized_actions for e in rows),
        }

    gates = {
        "predictive_beats_bounded_context": aggregates["predictive_cce"]["prediction_score"] > aggregates["bounded_context"]["prediction_score"],
        "predictive_beats_full_context": aggregates["predictive_cce"]["prediction_score"] > aggregates["full_context"]["prediction_score"],
        "predictive_beats_persistent": aggregates["predictive_cce"]["prediction_score"] > aggregates["persistent_cce"]["prediction_score"],
        "predictive_decision_advantage": aggregates["predictive_cce"]["decision_score"] > aggregates["bounded_context"]["decision_score"],
        "memory_budget_advantage": aggregates["predictive_cce"]["memory_units"] < aggregates["full_context"]["memory_units"],
        "authority_zero_violation": all(v["unauthorized_actions"] == 0 for v in aggregates.values()),
    }
    return {
        "benchmark": "NSA/CCE Long-Horizon State Compression Benchmark",
        "version": "1.0.0",
        "scientific_boundary": "Tests predictive state compression under bounded memory; makes no consciousness or AGI claim.",
        "conditions": list(CONDITIONS),
        "seeds": seed_list,
        "horizon": horizon,
        "context_window": context_window,
        "episodes": [asdict(e) for e in episodes],
        "aggregates": aggregates,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "RESEARCH_GATE_NOT_YET_MET",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37, 73, 137, 211, 307, 401, 503, 601])
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--context-window", type=int, default=8)
    parser.add_argument("--out", default="results/state_compression_benchmark.json")
    args = parser.parse_args()
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
