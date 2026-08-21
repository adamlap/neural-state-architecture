"""Deterministic bootstrap uncertainty analysis for Phase 19 improvements."""
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
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def analyze(values: list[float], *, samples: int = 10000, seed: int = 0) -> dict:
    numeric = [float(x) for x in values]
    if not numeric or not all(x == x and abs(x) != float("inf") for x in numeric):
        raise ValueError("improvements must be a non-empty finite numeric list")
    lo, hi = bootstrap_mean_ci(numeric, samples=samples, seed=seed)
    mean_value = sum(numeric) / len(numeric)
    return {
        "evaluations": len(numeric),
        "mean_improvement": mean_value,
        "bootstrap_95ci_low": lo,
        "bootstrap_95ci_high": hi,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "fraction_positive": sum(x > 0 for x in numeric) / len(numeric),
        "ci_excludes_zero": lo > 0 or hi < 0,
        "interpretation": "descriptive uncertainty only; independence and sampling assumptions must be justified before generalization",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSON containing an improvements array")
    parser.add_argument("--output", default="results/self-model-bootstrap.json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload["improvements"], samples=args.samples, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
