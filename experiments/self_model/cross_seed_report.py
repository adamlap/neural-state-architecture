"""Aggregate observed Phase 19 self-model evaluation JSON files.

This tool performs no model inference and creates no synthetic trajectories. It
only aggregates evaluation artifacts produced by an explicit trajectory
collector/evaluator. A predictor is considered a win only when its MSE is
strictly below the matched persistence baseline MSE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REQUIRED = ("predictor_mse", "persistence_mse")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def load_evaluation(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be an object")
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing required metrics: {', '.join(missing)}")
    return {key: _finite_number(payload[key], key) for key in REQUIRED}


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one evaluation file is required")
    rows = [load_evaluation(path) for path in paths]
    improvements = [row["persistence_mse"] - row["predictor_mse"] for row in rows]
    predictor_wins = [value > 0 for value in improvements]
    return {
        "files": [str(path) for path in paths],
        "count": len(rows),
        "predictor_mse_mean": mean(row["predictor_mse"] for row in rows),
        "predictor_mse_std": pstdev(row["predictor_mse"] for row in rows),
        "persistence_mse_mean": mean(row["persistence_mse"] for row in rows),
        "persistence_mse_std": pstdev(row["persistence_mse"] for row in rows),
        "improvement_mean": mean(improvements),
        "improvement_std": pstdev(improvements),
        "positive_improvement_fraction": sum(predictor_wins) / len(rows),
        "predictor_win_fraction": sum(predictor_wins) / len(rows),
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = aggregate(args.paths)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
