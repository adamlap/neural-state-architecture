"""Capacity sweep for the Holographic State Field experiment.

Run:
    python -m experiments.holographic_state_field.benchmark --out results/hsf_v1.json

The explicit vector-slot baseline stores one spatial key code and one value
code per association. HSF regenerates key patterns and stores a fixed field
plus a fixed value codebook. Byte counts are NumPy payload bytes, not RSS.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .field import HolographicStateField

DEFAULT_SEEDS = (7, 17, 37, 73, 137, 211, 307, 401, 503, 601)
DEFAULT_COUNTS = (8, 16, 32, 64, 128, 256, 512)
VOCABULARY = tuple(f"VALUE_{i}" for i in range(8))


def run_capacity(seed: int, count: int, *, height: int, width: int, channels: int) -> dict:
    field = HolographicStateField(
        VOCABULARY, height=height, width=width, channels=channels, seed=seed
    )
    records = [
        (f"ENTITY_{i:05d}", VOCABULARY[(i * 7 + seed) % len(VOCABULARY)])
        for i in range(count)
    ]
    for key, value in records:
        field.write(key, value)
    correct = sum(field.retrieve(key).value == value for key, value in records)
    field_bytes = field.memory_bytes
    slot_bytes = count * (height * width + channels) * 4
    return {
        "seed": seed,
        "memories": count,
        "retrieval_accuracy": correct / count,
        "correct": correct,
        "field_shape": list(field.shape),
        "field_bytes": field_bytes,
        "explicit_vector_slot_bytes": slot_bytes,
        "field_to_slot_memory_ratio": field_bytes / slot_bytes,
        "field_energy": field.field_energy,
    }


def run(seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS, *, height=32, width=32, channels=64):
    rows = [
        run_capacity(seed, count, height=height, width=width, channels=channels)
        for count in counts for seed in seeds
    ]
    aggregates = []
    for count in counts:
        group = [row for row in rows if row["memories"] == count]
        aggregates.append({
            "memories": count,
            "mean_retrieval_accuracy": statistics.fmean(r["retrieval_accuracy"] for r in group),
            "min_retrieval_accuracy": min(r["retrieval_accuracy"] for r in group),
            "mean_field_bytes": statistics.fmean(r["field_bytes"] for r in group),
            "mean_explicit_slot_bytes": statistics.fmean(r["explicit_vector_slot_bytes"] for r in group),
            "mean_memory_ratio": statistics.fmean(r["field_to_slot_memory_ratio"] for r in group),
        })
    return {
        "benchmark": "NSA Holographic State Field",
        "version": "0.1.0",
        "hypothesis": "A fixed multidimensional field can retrieve overlapping key/value associations by matched-filter reconstruction.",
        "scientific_boundary": "Synthetic associative recall only; not a test of language understanding, transformer attention replacement, or hardware efficiency.",
        "configuration": {
            "seeds": list(seeds), "counts": list(counts), "height": height,
            "width": width, "channels": channels, "vocabulary": list(VOCABULARY),
            "dtype": "float32",
        },
        "aggregates": aggregates,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--out", default="results/hsf_v1.json")
    args = parser.parse_args()
    report = run(args.seeds, args.counts, height=args.height, width=args.width, channels=args.channels)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["aggregates"]:
        print(f"n={row['memories']:>4} accuracy={row['mean_retrieval_accuracy']:.3f} "
              f"min={row['min_retrieval_accuracy']:.3f} field/slots={row['mean_memory_ratio']:.3f}")
    print(f"artifact={out}")


if __name__ == "__main__":
    main()
