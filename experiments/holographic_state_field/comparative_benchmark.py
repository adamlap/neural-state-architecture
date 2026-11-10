"""Controlled HSF vs HRR vs HDC vs exact-dictionary benchmark."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
from .baselines import ExactDictionary, HDCMemory, HRRMemory
from .field import HolographicStateField

DEFAULT_SEEDS = (7, 17, 37, 73, 137)
DEFAULT_COUNTS = (8, 16, 32, 64, 128, 256, 512)
VOCABULARY = tuple(f"VALUE_{i}" for i in range(8))

def _records(seed: int, count: int) -> list[tuple[str, str]]:
    return [(f"ENTITY_{i:05d}", VOCABULARY[(i * 7 + seed) % len(VOCABULARY)])
            for i in range(count)]

def _evaluate(store, records):
    correct = sum(store.retrieve(key).value == value for key, value in records)
    return correct / len(records), correct

def run_one(seed: int, count: int, *, height: int, width: int, channels: int) -> dict:
    records = _records(seed, count)
    dimension = height * width * channels
    stores = {
        "hsf": HolographicStateField(VOCABULARY, height=height, width=width,
                                     channels=channels, seed=seed),
        "hrr": HRRMemory(VOCABULARY, dimension=dimension, seed=seed),
        "hdc": HDCMemory(VOCABULARY, dimension=dimension, seed=seed),
        "exact_dictionary": ExactDictionary(),
    }
    for key, value in records:
        for store in stores.values():
            store.write(key, value)
    results = {}
    for name, store in stores.items():
        accuracy, correct = _evaluate(store, records)
        results[name] = {"accuracy": accuracy, "correct": correct,
                         "numeric_payload_bytes": store.memory_bytes}
    explicit_slot_bytes = count * (2 * dimension * 4)
    for name in ("hsf", "hrr", "hdc"):
        results[name]["explicit_vector_slot_bytes"] = explicit_slot_bytes
        results[name]["payload_to_explicit_slots_ratio"] = (
            results[name]["numeric_payload_bytes"] / explicit_slot_bytes)
    return {"seed": seed, "memories": count, "dimension": dimension,
            "field_shape": [height, width, channels], "methods": results}

def run(seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS, *, height=16, width=16, channels=8):
    runs = [run_one(seed, count, height=height, width=width, channels=channels)
            for count in counts for seed in seeds]
    aggregates = []
    for count in counts:
        group = [r for r in runs if r["memories"] == count]
        summary = {"memories": count}
        for method in ("hsf", "hrr", "hdc", "exact_dictionary"):
            values = [r["methods"][method]["accuracy"] for r in group]
            summary[method] = {
                "mean_accuracy": statistics.fmean(values),
                "minimum_accuracy": min(values),
                "mean_numeric_payload_bytes": statistics.fmean(
                    r["methods"][method]["numeric_payload_bytes"] for r in group)}
        aggregates.append(summary)
    return {
        "benchmark": "NSA distributed associative memory comparison",
        "version": "1.0.0",
        "question": "How do HSF, HRR and HDC compare on synthetic closed-vocabulary recall?",
        "scientific_boundary": "This does not test semantic generalisation, biological fidelity, language quality, hardware efficiency, or replacement of transformer attention.",
        "accounting": {
            "persistent_payload": "NumPy numeric payload only; excludes Python objects, strings, allocator overhead, and process RSS.",
            "explicit_slots": "Illustrative 2*D float32 values per record; not a measured Python dictionary footprint.",
            "dimension": height * width * channels,
            "hsf_shape": [height, width, channels],
            "hrr_hdc_dimension": height * width * channels,
            "caveat": "HSF tensor factorisation and flat HRR/HDC vectors have different inductive biases and compute costs. Equal flattened dimensions do not make algorithms mathematically identical."
        },
        "configuration": {"seeds": list(seeds), "counts": list(counts),
                          "vocabulary": list(VOCABULARY), "height": height,
                          "width": width, "channels": channels, "dtype": "float32 storage"},
        "aggregates": aggregates, "runs": runs}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out", default="results/hsf_comparison_v1.json")
    args = parser.parse_args()
    report = run(args.seeds, args.counts, height=args.height,
                 width=args.width, channels=args.channels)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["aggregates"]:
        print(f"n={row['memories']:>4}", end=" ")
        for method in ("hsf", "hrr", "hdc", "exact_dictionary"):
            print(f"{method}={row[method]['mean_accuracy']:.3f}", end=" ")
        print()
    print(f"artifact={output}")

if __name__ == "__main__":
    main()
