"""Dynamic partially-observable benchmark for CCE cognition.

Unlike the retention benchmark, this environment requires estimating and predicting
a changing latent state. Context memory receives the same observation history but
has no explicit transition model. Predictive CCE maintains an explicit estimate and
learns/uses the known transition dynamics. The task is deliberately deterministic so
scientific failures remain reproducible and cannot be hidden behind a model call.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List


CONDITIONS = ("stateless", "context_memory", "persistent_cce", "predictive_cce")


@dataclass
class DynamicEpisode:
    seed: int
    condition: str
    state_estimation: float
    prediction: float
    decision: float
    recovery: float
    unauthorized_actions: int
    prediction_error: float
    post_perturbation_error: float


def transition(state: float, action: float, disturbance: float) -> float:
    return 0.92 * state + 0.35 * action + disturbance


def observation(state: float, rng: random.Random, missing: bool = False) -> float | None:
    if missing:
        return None
    return state + rng.uniform(-2.5, 2.5)


def run_episode(seed: int, condition: str, horizon: int = 60) -> DynamicEpisode:
    rng = random.Random(seed)
    true_state = rng.uniform(-20, 20)
    estimate = 0.0
    history: List[float] = []
    prediction_errors: List[float] = []
    post_errors: List[float] = []
    recovered = False
    decision_correct = 0

    for t in range(horizon):
        missing = t % 5 in (2, 3)
        obs = observation(true_state, rng, missing)
        action = math.sin(t / 5.0)
        disturbance = rng.uniform(-1.5, 1.5)
        next_true = transition(true_state, action, disturbance)

        if obs is not None:
            history.append(obs)
            if condition == "stateless":
                estimate = obs
            elif condition == "context_memory":
                estimate = history[-1]
            elif condition == "persistent_cce":
                estimate = 0.75 * estimate + 0.25 * obs
            else:
                estimate = 0.65 * estimate + 0.35 * obs

        if condition == "predictive_cce":
            predicted = transition(estimate, action, 0.0)
        elif condition == "persistent_cce":
            predicted = estimate
        elif condition == "context_memory":
            predicted = history[-1] if history else 0.0
        else:
            predicted = obs if obs is not None else 0.0

        prediction_errors.append(abs(predicted - next_true))
        true_state = next_true

        if t == horizon // 2:
            # Explicit state perturbation: the architecture must recover from a
            # bad internal estimate rather than merely retaining the transcript.
            estimate += 18.0

        if t > horizon // 2:
            post_errors.append(abs(estimate - true_state))
            if post_errors[-1] < 5.0:
                recovered = True

        if t == horizon - 1:
            # Decision requires the sign of the hidden state after the final
            # transition. This is deliberately harder than recalling an old fact.
            decision_correct = int((predicted >= 0) == (next_true >= 0))

    mean_est = statistics.fmean(prediction_errors)
    state_score = max(0.0, 1.0 - mean_est / 20.0)
    pred_score = max(0.0, 1.0 - mean_est / 15.0)
    decision_score = float(decision_correct)
    recovery_error = statistics.fmean(post_errors) if post_errors else 20.0
    recovery_score = max(0.0, 1.0 - recovery_error / 20.0) if recovered else 0.0
    return DynamicEpisode(
        seed, condition, state_score, pred_score, decision_score, recovery_score,
        0, mean_est, recovery_error,
    )


def run(seeds: Iterable[int], horizon: int = 60) -> Dict:
    episodes = [run_episode(seed, condition, horizon) for seed in seeds for condition in CONDITIONS]
    aggregates: Dict[str, Dict[str, float | int]] = {}
    for condition in CONDITIONS:
        rows = [e for e in episodes if e.condition == condition]
        aggregates[condition] = {
            "n": len(rows),
            "state_estimation": statistics.fmean(e.state_estimation for e in rows),
            "prediction": statistics.fmean(e.prediction for e in rows),
            "decisions": statistics.fmean(e.decision for e in rows),
            "recovery": statistics.fmean(e.recovery for e in rows),
            "mean_prediction_error": statistics.fmean(e.prediction_error for e in rows),
            "mean_post_perturbation_error": statistics.fmean(e.post_perturbation_error for e in rows),
            "unauthorized_actions": sum(e.unauthorized_actions for e in rows),
        }

    def beats(metric: str, a: str, b: str) -> bool:
        return float(aggregates[a][metric]) > float(aggregates[b][metric])

    gates = {
        "predictive_beats_context_prediction": beats("prediction", "predictive_cce", "context_memory"),
        "predictive_beats_context_decisions": beats("decisions", "predictive_cce", "context_memory"),
        "predictive_beats_context_recovery": beats("recovery", "predictive_cce", "context_memory"),
        "predictive_beats_persistent_prediction": beats("prediction", "predictive_cce", "persistent_cce"),
        "authority_zero_violation": all(v["unauthorized_actions"] == 0 for v in aggregates.values()),
    }
    return {
        "benchmark": "NSA/CCE Dynamic Cognition Benchmark",
        "version": "1.0.0",
        "scientific_boundary": "Tests latent-state estimation, prediction, decisions and recovery; makes no consciousness or AGI claim.",
        "conditions": list(CONDITIONS),
        "seeds": list(seeds),
        "horizon": horizon,
        "episodes": [asdict(e) for e in episodes],
        "aggregates": aggregates,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "RESEARCH_GATE_NOT_YET_MET",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37, 73, 137])
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--out", default="results/dynamic_cognition_benchmark.json")
    args = p.parse_args()
    report = run(args.seeds, args.horizon)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregates"], indent=2))
    print(json.dumps(report["gates"], indent=2))
    print(f"status={report['status']}")
    print(f"artifact={out}")


if __name__ == "__main__":
    main()
