"""Audit-only decomposition of HSF retrieval interference."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .correlated_key_benchmark import (
    DEFAULT_COUNTS, DEFAULT_SEEDS, DEFAULT_SIMILARITIES,
    FAMILIES, VOCABULARY, correlated_cues,
)
from .field import HolographicStateField


def _records(names, seed):
    return [
        (name, VOCABULARY[((i // FAMILIES) * 3 + (i % FAMILIES) * 5 + seed)
                          % len(VOCABULARY)], i % FAMILIES)
        for i, name in enumerate(names)
    ]


def _scores(vector, codes):
    return {value: float(np.dot(vector, code)) for value, code in codes.items()}


def _margin(scores, expected):
    ordered = sorted(scores.values(), reverse=True)
    return scores[expected] - (ordered[1] if len(ordered) > 1 else 0.0)


def run(*, seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS,
        similarities=DEFAULT_SIMILARITIES, height=16, width=16, channels=16):
    if not seeds or not counts or not similarities:
        raise ValueError("seeds, counts, and similarities must be non-empty")
    if any(int(n) < FAMILIES for n in counts):
        raise ValueError(f"counts must be >= {FAMILIES}")
    if min(height, width, channels) <= 0 or channels < len(VOCABULARY):
        raise ValueError("invalid field dimensions or insufficient channels")
    if any(not 0 <= rho < 1 for rho in similarities):
        raise ValueError("similarities must be in [0, 1)")

    runs = []
    for rho in similarities:
        for count in counts:
            for seed in seeds:
                names, cues, realised = correlated_cues(
                    seed, height * width, count, rho
                )
                records = _records(names, seed)
                total = HolographicStateField(
                    VOCABULARY, height=height, width=width,
                    channels=channels, seed=seed,
                )
                family_fields = {
                    f: HolographicStateField(
                        VOCABULARY, height=height, width=width,
                        channels=channels, seed=seed,
                    ) for f in range(FAMILIES)
                }
                targets = {}
                for name, value, family in records:
                    total.write_pattern(cues[name], value)
                    family_fields[family].write_pattern(cues[name], value)
                    one = HolographicStateField(
                        VOCABULARY, height=height, width=width,
                        channels=channels, seed=seed,
                    )
                    one.write_pattern(cues[name], value)
                    targets[name] = one._field

                codes = total._value_codes
                pairs = [
                    float(np.dot(codes[a], codes[b]))
                    for i, a in enumerate(VOCABULARY)
                    for b in VOCABULARY[i + 1:]
                ]
                per_query = []
                for name, expected, family in records:
                    query = cues[name].reshape(height, width)
                    target = np.einsum("hw,hwc->c", query, targets[name])
                    same = np.einsum(
                        "hw,hwc->c", query,
                        family_fields[family]._field - targets[name],
                    )
                    cross = np.einsum(
                        "hw,hwc->c", query,
                        total._field - family_fields[family]._field,
                    )
                    vectors = {
                        "target_only": target,
                        "target_plus_same_family": target + same,
                        "target_plus_cross_family": target + cross,
                        "full": target + same + cross,
                    }
                    scores = {k: _scores(v, codes) for k, v in vectors.items()}
                    per_query.append({
                        "accuracy": {
                            k: int(max(s, key=s.get) == expected)
                            for k, s in scores.items()
                        },
                        "margin": {k: _margin(s, expected) for k, s in scores.items()},
                        "target_norm": float(np.linalg.norm(target)),
                        "same_norm": float(np.linalg.norm(same)),
                        "cross_norm": float(np.linalg.norm(cross)),
                        "same_target_projection": float(np.dot(same, codes[expected])),
                        "cross_target_projection": float(np.dot(cross, codes[expected])),
                    })

                labels = ("target_only", "target_plus_same_family",
                          "target_plus_cross_family", "full")
                runs.append({
                    "seed": seed, "memories": count,
                    "target_within_family_cosine": rho,
                    "mean_realised_within_family_cosine": realised,
                    "component_accuracy": {
                        k: statistics.fmean(x["accuracy"][k] for x in per_query)
                        for k in labels
                    },
                    "mean_component_margin": {
                        k: statistics.fmean(x["margin"][k] for x in per_query)
                        for k in labels
                    },
                    "same_family_interference_to_signal": statistics.fmean(
                        x["same_norm"] / max(x["target_norm"], 1e-12)
                        for x in per_query
                    ),
                    "cross_family_interference_to_signal": statistics.fmean(
                        x["cross_norm"] / max(x["target_norm"], 1e-12)
                        for x in per_query
                    ),
                    "same_family_target_code_projection": statistics.fmean(
                        x["same_target_projection"] for x in per_query
                    ),
                    "cross_family_target_code_projection": statistics.fmean(
                        x["cross_target_projection"] for x in per_query
                    ),
                    "value_codebook_mean_pairwise_cosine": (
                        statistics.fmean(pairs) if pairs else 0.0
                    ),
                    "value_codebook_max_abs_pairwise_cosine": max(
                        (abs(x) for x in pairs), default=0.0
                    ),
                })

    labels = ("target_only", "target_plus_same_family",
              "target_plus_cross_family", "full")
    aggregates = []
    for rho in similarities:
        for count in counts:
            group = [x for x in runs if x["target_within_family_cosine"] == rho
                     and x["memories"] == count]
            aggregates.append({
                "target_within_family_cosine": rho, "memories": count,
                "component_accuracy": {
                    k: statistics.fmean(x["component_accuracy"][k] for x in group)
                    for k in labels
                },
                "mean_component_margin": {
                    k: statistics.fmean(x["mean_component_margin"][k] for x in group)
                    for k in labels
                },
                "same_family_interference_to_signal": statistics.fmean(
                    x["same_family_interference_to_signal"] for x in group
                ),
                "cross_family_interference_to_signal": statistics.fmean(
                    x["cross_family_interference_to_signal"] for x in group
                ),
                "same_family_target_code_projection": statistics.fmean(
                    x["same_family_target_code_projection"] for x in group
                ),
                "cross_family_target_code_projection": statistics.fmean(
                    x["cross_family_target_code_projection"] for x in group
                ),
                "value_codebook_mean_pairwise_cosine": statistics.fmean(
                    x["value_codebook_mean_pairwise_cosine"] for x in group
                ),
                "value_codebook_max_abs_pairwise_cosine": max(
                    x["value_codebook_max_abs_pairwise_cosine"] for x in group
                ),
            })
    return {
        "benchmark": "HSF interference-source decomposition",
        "version": "1.0.0",
        "scientific_boundary": (
            "Synthetic closed-vocabulary diagnostic. Component fields exist "
            "only in this audit harness; HSF runtime is unchanged."
        ),
        "configuration": {
            "seeds": list(seeds), "counts": list(counts),
            "target_similarities": list(similarities),
            "field_shape": [height, width, channels],
            "families": FAMILIES, "vocabulary": list(VOCABULARY),
        },
        "component_definitions": {
            "target_only": "Queried association only.",
            "target_plus_same_family": "Target plus other keys in its correlated family.",
            "target_plus_cross_family": "Target plus keys in other families.",
            "full": "All associations; equivalent to normal HSF retrieval.",
        },
        "aggregates": aggregates, "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--similarities", nargs="+", type=float,
                        default=list(DEFAULT_SIMILARITIES))
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    args = parser.parse_args()
    report = run(seeds=args.seeds, counts=args.counts,
                 similarities=args.similarities, height=args.height,
                 width=args.width, channels=args.channels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"runs={len(report['runs'])} aggregates={len(report['aggregates'])}")


if __name__ == "__main__":
    main()
