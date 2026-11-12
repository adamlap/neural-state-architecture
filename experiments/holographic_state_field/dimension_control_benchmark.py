"""Dimension-controlled HSF / HRR / HDC correlated-key diagnostic.

Two comparisons are reported separately:
* key_dimension_matched: HRR/HDC use the same key dimension as HSF (H*W).
  Their value code dimension consequently equals that key dimension.
* numeric_payload_matched: HRR/HDC dimensions are chosen so their float32
  state payload exactly matches the HSF field plus its fixed value codebook.

Neither arm is presented as a fully capacity- and compute-equivalent test.
The report makes key, value, and persistent numeric payload dimensions
explicit so results cannot be mistaken for a single perfectly fair contest.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .baselines import HDCMemory, HRRMemory
from .correlated_key_benchmark import FAMILIES, VOCABULARY, correlated_cues
from .field import HolographicStateField

DEFAULT_SEEDS = (7, 17, 37, 73, 137)
DEFAULT_COUNTS = (16, 32, 64, 128)
DEFAULT_SIMILARITIES = (0.0, 0.5, 0.75)
DEFAULT_SHAPE = (16, 16, 16)


def _records(names: list[str], seed: int) -> list[tuple[str, str]]:
    return [
        (name, VOCABULARY[((index // FAMILIES) * 3 + (index % FAMILIES) * 5 + seed)
                          % len(VOCABULARY)])
        for index, name in enumerate(names)
    ]


def _run_hsf(seed: int, count: int, rho: float,
             height: int, width: int, channels: int) -> dict:
    key_dim = height * width
    names, cues, realised = correlated_cues(seed, key_dim, count, rho)
    store = HolographicStateField(
        VOCABULARY, height=height, width=width, channels=channels, seed=seed
    )
    records = _records(names, seed)
    for name, value in records:
        store.write_pattern(cues[name], value)
    correct = sum(store.retrieve_pattern(cues[name]).value == value
                  for name, value in records)
    return {
        "accuracy": correct / count,
        "correct": correct,
        "key_dimension": key_dim,
        "value_dimension": channels,
        "field_elements": key_dim * channels,
        "numeric_payload_bytes": store.memory_bytes,
        "mean_realised_within_family_cosine": realised,
    }


def _run_vector(method: str, seed: int, count: int, rho: float,
                dimension: int) -> dict:
    names, cues, realised = correlated_cues(seed, dimension, count, rho)
    memory_type = {"hrr": HRRMemory, "hdc": HDCMemory}[method]
    store = memory_type(VOCABULARY, dimension=dimension, seed=seed)
    records = _records(names, seed)
    for name, value in records:
        store.write_vector(cues[name], value)
    correct = sum(store.retrieve_vector(cues[name]).value == value
                  for name, value in records)
    return {
        "accuracy": correct / count,
        "correct": correct,
        "key_dimension": dimension,
        "value_dimension": dimension,
        "field_elements": dimension,
        "numeric_payload_bytes": store.memory_bytes,
        "mean_realised_within_family_cosine": realised,
    }


def run(*, seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS,
        similarities=DEFAULT_SIMILARITIES, height=16, width=16, channels=16):
    if not seeds or not counts or not similarities:
        raise ValueError("seeds, counts, and similarities must be non-empty")
    if any(int(count) < 1 for count in counts):
        raise ValueError("counts must be positive")
    if min(height, width, channels) <= 0:
        raise ValueError("field dimensions must be positive")
    if any(not 0.0 <= rho < 1.0 for rho in similarities):
        raise ValueError("similarities must be in [0, 1)")
    if channels < len(VOCABULARY):
        raise ValueError("channels must be >= vocabulary size")

    key_dim = height * width
    field_elements = key_dim * channels
    arms = {
        "key_dimension_matched": {
            "description": "All methods use the HSF spatial key dimension; vector baselines also use that dimension for values.",
            "baseline_dimension": key_dim,
        },
        "numeric_payload_matched": {
            "description": "Vector baselines use a dimension chosen to exactly match HSF field plus fixed-codebook float32 payload bytes.",
            "baseline_dimension": field_elements + len(VOCABULARY) * channels,
        },
    }
    runs = []
    for rho in similarities:
        for count in counts:
            for seed in seeds:
                hsf = _run_hsf(seed, count, rho, height, width, channels)
                for arm, metadata in arms.items():
                    baseline_dim = metadata["baseline_dimension"]
                    for method in ("hrr", "hdc"):
                        baseline = _run_vector(method, seed, count, rho, baseline_dim)
                        runs.append({
                            "arm": arm, "method": method, "seed": seed,
                            "memories": count, "target_within_family_cosine": rho,
                            **baseline,
                        })
                    runs.append({
                        "arm": arm, "method": "hsf", "seed": seed,
                        "memories": count, "target_within_family_cosine": rho,
                        **hsf,
                    })

    aggregates = []
    for arm in arms:
        for rho in similarities:
            for count in counts:
                for method in ("hsf", "hrr", "hdc"):
                    group = [row for row in runs if row["arm"] == arm
                             and row["method"] == method
                             and row["target_within_family_cosine"] == rho
                             and row["memories"] == count]
                    aggregates.append({
                        "arm": arm, "method": method,
                        "target_within_family_cosine": rho, "memories": count,
                        "mean_accuracy": statistics.fmean(r["accuracy"] for r in group),
                        "minimum_accuracy": min(r["accuracy"] for r in group),
                        "key_dimension": group[0]["key_dimension"],
                        "value_dimension": group[0]["value_dimension"],
                        "numeric_payload_bytes": group[0]["numeric_payload_bytes"],
                        "mean_realised_within_family_cosine": statistics.fmean(
                            r["mean_realised_within_family_cosine"] for r in group
                        ),
                    })

    return {
        "benchmark": "HSF dimension-controlled correlated-key diagnostic",
        "version": "1.0.0",
        "scientific_boundary": (
            "Synthetic closed-vocabulary associative recall only. This is a "
            "dimension diagnostic, not a complete equivalence of compute, "
            "capacity, code geometry, or runtime overhead."
        ),
        "configuration": {
            "seeds": list(seeds), "counts": list(counts),
            "target_similarities": list(similarities),
            "field_shape": [height, width, channels],
            "hsf_key_dimension": key_dim,
            "hsf_value_dimension": channels,
            "hsf_field_elements": field_elements,
            "vocabulary": list(VOCABULARY),
        },
        "comparison_arms": arms,
        "interpretation": (
            "Read each arm independently. The key-dimension arm aligns cue "
            "length but not value-code length or persistent bytes. The "
            "numeric-payload arm aligns dominant float32 payload but not cue "
            "length or value-code length. HSF's fixed value codebook is "
            "included in its payload bytes."
        ),
        "aggregates": aggregates,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--similarities", nargs="+", type=float,
                        default=list(DEFAULT_SIMILARITIES))
    parser.add_argument("--height", type=int, default=DEFAULT_SHAPE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_SHAPE[1])
    parser.add_argument("--channels", type=int, default=DEFAULT_SHAPE[2])
    args = parser.parse_args()
    report = run(seeds=args.seeds, counts=args.counts,
                 similarities=args.similarities, height=args.height,
                 width=args.width, channels=args.channels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "runs": len(report["runs"]),
        "aggregates": len(report["aggregates"]),
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
