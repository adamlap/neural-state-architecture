"""Aggregate independent Phase 19 predictor evaluations.

Each input JSON is produced by ``train_live_trajectory.py`` from a trajectory
collected at the Ollama boundary. This module deliberately aggregates observed
results; it does not manufacture trajectories or infer hidden model state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def load_result(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "seed",
        "test_predictor_mse",
        "test_persistence_mse",
        "test_mse_improvement",
        "predictor_beats_persistence",
        "finite",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if not value["finite"]:
        raise ValueError(f"{path}: predictor contains non-finite parameters")
    for key in ("test_predictor_mse", "test_persistence_mse", "test_mse_improvement"):
        if not math.isfinite(float(value[key])):
            raise ValueError(f"{path}: {key} is non-finite")
    return value


def aggregate(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one evaluation is required")
    improvements = [float(row["test_mse_improvement"]) for row in results]
    predictor_mse = [float(row["test_predictor_mse"]) for row in results]
    persistence_mse = [float(row["test_persistence_mse"]) for row in results]
    wins = [bool(row["predictor_beats_persistence"]) for row in results]
    return {
        "evaluations": len(results),
        "seeds": [int(row["seed"]) for row in results],
        "mean_predictor_mse": mean(predictor_mse),
        "std_predictor_mse": pstdev(predictor_mse) if len(results) > 1 else 0.0,
        "mean_persistence_mse": mean(persistence_mse),
        "std_persistence_mse": pstdev(persistence_mse) if len(results) > 1 else 0.0,
        "mean_mse_improvement": mean(improvements),
        "std_mse_improvement": pstdev(improvements) if len(results) > 1 else 0.0,
        "positive_improvement_fraction": sum(x > 0 for x in improvements) / len(results),
        "predictor_win_fraction": sum(wins) / len(results),
        "all_finite": all(bool(row["finite"]) for row in results),
        "scientific_boundary": "explicit NSA self-state only; no transformer hidden-state access",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output", default="results/live-self-model-multiseed.json")
    args = parser.parse_args()

    results = [load_result(path) for path in args.evaluations]
    summary = aggregate(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
