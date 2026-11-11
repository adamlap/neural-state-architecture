"""HSF factorisation sweep under a fixed total field element budget.

The sweep varies the split between spatial-key dimension (H*W) and value
channel dimension (C), while keeping H*W*C constant. HRR/HDC use the full
flattened dimension for both key and value codes. Results report those
dimensions explicitly; this is a diagnostic frontier, not a claim of
perfectly matched compute or representational capacity.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .correlated_key_benchmark import (
    DEFAULT_COUNTS,
    DEFAULT_SEEDS,
    DEFAULT_SIMILARITIES,
    _run_method,
)

DEFAULT_SHAPES = ((16, 16, 32), (16, 8, 64), (8, 8, 128), (8, 16, 64))


def run(*, seeds=DEFAULT_SEEDS, counts=(16, 32, 64, 128),
        similarities=(0.0, 0.5, 0.75), shapes=DEFAULT_SHAPES):
    """Run all HSF factorizations and flat baselines at equal field element count."""
    total_dimensions = {height * width * channels for height, width, channels in shapes}
    if len(total_dimensions) != 1:
        raise ValueError("all shapes must have the same H*W*C element budget")
    if any(channels < 16 for _, _, channels in shapes):
        raise ValueError("channels must be >= 16 for the eight-value vocabulary")
    total_dimension = next(iter(total_dimensions))
    runs = []
    for height, width, channels in shapes:
        for rho in similarities:
            for count in counts:
                for seed in seeds:
                    result = _run_method("hsf", seed, count, rho, height, width, channels)
                    result.update({
                        "method": "hsf",
                        "seed": seed,
                        "memories": count,
                        "target_within_family_cosine": rho,
                        "field_shape": [height, width, channels],
                        "total_dimensions": total_dimension,
                        "key_dimension": height * width,
                        "value_dimension": channels,
                    })
                    runs.append(result)
    # Flat baselines appear once per seed/load/similarity because their shape
    # does not change with the HSF factorisation.
    baseline_runs = []
    for rho in similarities:
        for count in counts:
            for seed in seeds:
                for method in ("hrr", "hdc"):
                    result = _run_method(method, seed, count, rho, *shapes[0])
                    result.update({
                        "method": method,
                        "seed": seed,
                        "memories": count,
                        "target_within_family_cosine": rho,
                        "field_shape": None,
                        "total_dimensions": total_dimension,
                        "key_dimension": total_dimension,
                        "value_dimension": total_dimension,
                    })
                    baseline_runs.append(result)
    runs.extend(baseline_runs)

    aggregates = []
    for rho in similarities:
        for count in counts:
            for height, width, channels in shapes:
                group = [r for r in runs if r["method"] == "hsf"
                         and r["field_shape"] == [height, width, channels]
                         and r["target_within_family_cosine"] == rho
                         and r["memories"] == count]
                aggregates.append({
                    "method": "hsf",
                    "field_shape": [height, width, channels],
                    "key_dimension": height * width,
                    "value_dimension": channels,
                    "target_within_family_cosine": rho,
                    "memories": count,
                    "mean_accuracy": statistics.fmean(r["accuracy"] for r in group),
                    "minimum_accuracy": min(r["accuracy"] for r in group),
                    "numeric_payload_bytes": group[0]["numeric_payload_bytes"],
                })
            for method in ("hrr", "hdc"):
                group = [r for r in runs if r["method"] == method
                         and r["target_within_family_cosine"] == rho
                         and r["memories"] == count]
                aggregates.append({
                    "method": method,
                    "field_shape": None,
                    "key_dimension": total_dimension,
                    "value_dimension": total_dimension,
                    "target_within_family_cosine": rho,
                    "memories": count,
                    "mean_accuracy": statistics.fmean(r["accuracy"] for r in group),
                    "minimum_accuracy": min(r["accuracy"] for r in group),
                    "numeric_payload_bytes": group[0]["numeric_payload_bytes"],
                })
    return {
        "benchmark": "HSF fixed-budget factorisation sweep",
        "version": "1.0.0",
        "total_dimensions": total_dimension,
        "seeds": list(seeds),
        "counts": list(counts),
        "similarities": list(similarities),
        "shapes": [list(shape) for shape in shapes],
        "interpretation": (
            "All HSF shapes have equal H*W*C field elements. Factorisation "
            "changes spatial cue dimension and value-code dimension. HRR/HDC "
            "use the full flat dimension for both. Numeric payload includes "
            "HSF's fixed value codebook; it excludes Python/runtime overhead. "
            "This isolates the allocation trade-off but does not equalise "
            "compute, key dimension, or value dimension."
        ),
        "runs": runs,
        "aggregates": aggregates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(result["runs"]), "aggregates": len(result["aggregates"]),
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
