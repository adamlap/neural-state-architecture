"""Deterministic bootstrap uncertainty analysis for Phase 19 improvements.

The input is a list of observed predictor-vs-persistence improvements. This
module never creates trajectories or model-state observations. It only quantifies
uncertainty around the supplied independent evaluations.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def bootstrap_mean_ci(values: list[float], *, samples: int = 10000, seed: int = 0) -> tuple[float, float]:
    if not values:
        raise ValueError("at least one improvement is required")
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def analyze(values: list[float], *, samples: int = 10000, seed: int = 0) -> dict:
    if not values:
        raise ValueError("at least one improvement is required")
    if not all(isinstance(x, (int, float)) for x in values):
        raise ValueError("improvements must be numeric")
    lo, hi = bootstrap_mean_ci([float(x) for x in values], samples=samples, seed=seed)
    mean_value = sum(values) / len(values)
    return {
        "evaluations": len(values),
        "mean_improvement": mean_value,
        "bootstrap_95ci_low": lo,
        "bootstrap_95ci_high": hi,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "fraction_positive": sum(x > 0 for x in values) / len(values),
        "ci_excludes_zero": lo > 0 or hi < 0,
        "interpretation": (
            "descriptive uncertainty only; independence and sampling assumptions "
            "must be justified before making a generalization"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON containing an improvements array")
    parser.add_argument("--output", default="results/self-model-bootstrap.json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    values = payload.get("improvements")
    if values is None:
        raise ValueError("input JSON must contain an 'improvements' array")
    result = analyze(values, samples=args.samples, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
