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

``persistent_cce`` and ``predictive_cce`` both use real Kalman filters (see
``_kalman.py``) instead of a hand-tuned ``0.7 * estimate + 0.3 * observation``
blend. ``persistent_cce`` filters each of the 3 dimensions independently with no
dynamics model (``a=1``); ``predictive_cce`` additionally knows the true
position/velocity coupling and the velocity control gain, so it can extrapolate
between observations of a given dimension instead of holding it constant.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from experiments.cognitive._kalman import ConstantVelocityKalman, ScalarKalman

CONDITIONS = ("stateless", "bounded_context", "full_context", "persistent_cce", "predictive_cce")

_OBSERVATION_STD = 0.4 / (3 ** 0.5)
_STEP_NOISE_VAR = (0.8 ** 2) / 3.0  # variance of uniform(-0.8, 0.8)
_BIAS_NOISE_VAR = (0.02 ** 2) * _STEP_NOISE_VAR


@dataclass
class Episode:
    seed: int
    condition: str
    mean_prediction_error: float
    prediction_score: float
    decision_score: float
    memory_units: int
    unauthorized_actions: int


def step(state: Tuple[float, float, float], action: float, noise: float) -> Tuple[float, float, float]:
    position, velocity, bias = state
    return (
        position + velocity,
        0.97 * velocity + 0.10 * action + noise,
        0.999 * bias + 0.02 * noise,
    )


def observe(state: Tuple[float, float, float], dimension: int, rng: random.Random) -> float:
    return state[dimension] + rng.uniform(-0.4, 0.4)


def run_episode(seed: int, condition: str, horizon: int = 200, context_window: int = 8) -> Episode:
    rng = random.Random(seed)
    state = (rng.uniform(-10, 10), rng.uniform(-1, 1), rng.uniform(-3, 3))
    history: List[Tuple[int, int, float]] = []
    errors: List[float] = []

    # persistent_cce: 3 independent Kalman filters, no dynamics/control model.
    # A naive "a=1, no dynamics" model still needs a realistic process-noise
    # estimate of how much an unmodelled quantity can drift between the sparse
    # observations of a given dimension (every 3rd step here). Position is a
    # literal running integral of velocity, so it can drift arbitrarily far in a
    # consistent direction; a filter with no dynamics model has no principled
    # basis to treat a direct new measurement as an "outlier" (that concept only
    # makes sense relative to a dynamics prediction it does not have), so
    # outlier rejection is disabled here rather than tuned to a magic threshold.
    persistent = [ScalarKalman(measurement_noise=_OBSERVATION_STD, process_noise=1.0, outlier_sigma=1e6) for _ in range(3)]
    # predictive_cce: position+velocity filtered jointly with the known coupling
    # and control gain; bias filtered with its known (near-unity) decay.
    pv = ConstantVelocityKalman(f11=1.0, f12=1.0, f21=0.0, f22=0.97, b2=0.10,
                                 process_noise=_STEP_NOISE_VAR, measurement_noise=_OBSERVATION_STD,
outlier_sigma=6.0)
    bias_filter = ScalarKalman(a=0.999, b=0.0, process_noise=_BIAS_NOISE_VAR,
                                measurement_noise=_OBSERVATION_STD, outlier_sigma=6.0)

    for t in range(horizon):
        dim = t % 3
        missing = t % 7 in (0, 1)
        obs = None if missing else observe(state, dim, rng)
        if obs is not None:
            history.append((t, dim, obs))

        # `obs` measures the *current* (pre-transition) state, so filters must be
        # updated with it before being predicted forward to forecast true_next.
        if condition == "persistent_cce" and obs is not None:
            persistent[dim].update(obs)
        elif condition == "predictive_cce" and obs is not None:
            if dim in (0, 1):
                pv.update_channel(obs, channel=dim)
            else:
                bias_filter.update(obs)

        if condition == "stateless":
            estimate = [0.0, 0.0, 0.0]
            if obs is not None:
                estimate[dim] = obs
        elif condition == "bounded_context":
            estimate = [0.0, 0.0, 0.0]
            for _, d, value in history[-context_window:]:
                estimate[d] = value
        elif condition == "full_context":
            estimate = [0.0, 0.0, 0.0]
            for _, d, value in history:
                estimate[d] = value
        elif condition == "persistent_cce":
            estimate = [f.x for f in persistent]
        else:
            estimate = [pv.position, pv.velocity, bias_filter.x]

        action = math.sin(t / 9.0)
        noise = rng.uniform(-0.8, 0.8)
        true_next = step(state, action, noise)

        if condition == "predictive_cce":
            # Known deterministic part of the transition (the noise term itself
            # is unpredictable regardless of estimator quality).
            predicted = [
                estimate[0] + estimate[1],
                0.97 * estimate[1] + 0.10 * action,
                0.999 * estimate[2],
            ]
        else:
            predicted = estimate[:]

        errors.append(sum(abs(a - b) for a, b in zip(predicted, true_next)) / 3.0)
        state = true_next

        if condition == "persistent_cce":
            for f in persistent:
                f.predict()
        elif condition == "predictive_cce":
            pv.predict(control=action)
            bias_filter.predict()

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
        "version": "2.0.0",
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
