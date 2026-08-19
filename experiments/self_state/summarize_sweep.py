"""Aggregate self-state perturbation sweep artifacts into reproducible evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


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
        by_perturbation.append(
            {
                "perturbation": perturbation,
                "samples": len(group),
                "mean_recovery_advantage": mean(recovery),
                "std_recovery_advantage": pstdev(recovery) if len(recovery) > 1 else 0.0,
                "positive_recovery_advantage_fraction": mean(v > 0.0 for v in recovery),
                "mean_auc_advantage": mean(auc),
                "std_auc_advantage": pstdev(auc) if len(auc) > 1 else 0.0,
                "positive_auc_advantage_fraction": mean(v > 0.0 for v in auc),
            }
        )

    recovery_all = [row["recovery_advantage"] for row in rows]
    auc_all = [row["auc_advantage"] for row in rows]
    return {
        "seeds": sorted(set(seeds)),
        "seed_count": len(set(seeds)),
        "perturbations": perturbations,
        "sample_count": len(rows),
        "mean_recovery_advantage": mean(recovery_all),
        "positive_recovery_advantage_fraction": mean(v > 0.0 for v in recovery_all),
        "mean_auc_advantage": mean(auc_all),
        "positive_auc_advantage_fraction": mean(v > 0.0 for v in auc_all),
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
