"""Aggregate self-state perturbation sweep artifacts into reproducible evidence."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI for the sample mean."""
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    rng = random.Random(seed)
    n = len(values)
    means = [mean(rng.choices(values, k=n)) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return _percentile(means, alpha), _percentile(means, 1.0 - alpha)


def _metric_summary(values: list[float]) -> dict:
    low, high = bootstrap_ci(values)
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "ci95": [low, high],
        "positive_fraction": mean(v > 0.0 for v in values),
    }


def summarize(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("no sweep artifacts supplied")

    rows: list[dict] = []
    seeds: list[int] = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result.get("finite") is True, f"non-finite result: {path}"
        seed = int(result["seed"])
        seeds.append(seed)
        for row in result["results"]:
            rows.append(
                {
                    "seed": seed,
                    "perturbation": float(row["perturbation"]),
                    "recovery_advantage": float(row["recovery_advantage"]),
                    "auc_advantage": float(row["auc_advantage"]),
                }
            )

    perturbations = sorted({row["perturbation"] for row in rows})
    by_perturbation = []
    for perturbation in perturbations:
        group = [row for row in rows if row["perturbation"] == perturbation]
        recovery = [row["recovery_advantage"] for row in group]
        auc = [row["auc_advantage"] for row in group]
        recovery_summary = _metric_summary(recovery)
        auc_summary = _metric_summary(auc)
        by_perturbation.append(
            {
                "perturbation": perturbation,
                "samples": len(group),
                "mean_recovery_advantage": recovery_summary["mean"],
                "std_recovery_advantage": recovery_summary["std"],
                "recovery_advantage_ci95": recovery_summary["ci95"],
                "positive_recovery_advantage_fraction": recovery_summary["positive_fraction"],
                "mean_auc_advantage": auc_summary["mean"],
                "std_auc_advantage": auc_summary["std"],
                "auc_advantage_ci95": auc_summary["ci95"],
                "positive_auc_advantage_fraction": auc_summary["positive_fraction"],
            }
        )

    recovery_all = [row["recovery_advantage"] for row in rows]
    auc_all = [row["auc_advantage"] for row in rows]
    recovery_summary = _metric_summary(recovery_all)
    auc_summary = _metric_summary(auc_all)
    return {
        "seeds": sorted(set(seeds)),
        "seed_count": len(set(seeds)),
        "perturbations": perturbations,
        "sample_count": len(rows),
        "mean_recovery_advantage": recovery_summary["mean"],
        "positive_recovery_advantage_fraction": recovery_summary["positive_fraction"],
        "recovery_advantage_ci95": recovery_summary["ci95"],
        "mean_auc_advantage": auc_summary["mean"],
        "positive_auc_advantage_fraction": auc_summary["positive_fraction"],
        "auc_advantage_ci95": auc_summary["ci95"],
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
            "method": "percentile bootstrap of the sample mean",
        },
        "by_perturbation": by_perturbation,
        "all_finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.paths)
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
