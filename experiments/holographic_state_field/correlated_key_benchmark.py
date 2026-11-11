"""Controlled correlated-key interference experiment for HSF, HRR, and HDC.

Keys in each family share a tunable fraction of their bipolar coordinates.
The expected within-family cosine similarity is rho. Each method receives
method-specific vectors generated from the same family/item seeds; dimensions
differ, so realised cosine distributions are reported rather than assumed equal.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import numpy as np
from .baselines import HDCMemory, HRRMemory
from .field import HolographicStateField

DEFAULT_SEEDS = (7, 17, 37, 73, 137)
DEFAULT_COUNTS = (8, 16, 32, 64, 128, 256)
DEFAULT_SIMILARITIES = (0.0, 0.25, 0.5, 0.75, 0.9)
VOCABULARY = tuple(f"VALUE_{i}" for i in range(8))
FAMILIES = 4


def correlated_cues(seed: int, dimension: int, count: int, rho: float) -> tuple[list[str], dict[str, np.ndarray], float]:
    """Return deterministic bipolar cues with target within-family cosine rho."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1); rho=1 would duplicate every family key")
    rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, count, int(rho * 10000)]))
    prototype = {
        family: rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
        for family in range(FAMILIES)
    }
    flip_probability = (1.0 - np.sqrt(rho)) / 2.0
    names: list[str] = []
    cues: dict[str, np.ndarray] = {}
    for index in range(count):
        family = index % FAMILIES
        name = f"FAMILY_{family}:ITEM_{index:05d}"
        flips = rng.random(dimension) < flip_probability
        cue = prototype[family].copy()
        cue[flips] *= -1.0
        cues[name] = cue / np.sqrt(np.float32(dimension))
        names.append(name)

    within = []
    for family in range(FAMILIES):
        members = [cues[name] for name in names if name.startswith(f"FAMILY_{family}:")]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                within.append(float(np.dot(members[i], members[j])))
    realised = statistics.fmean(within) if within else 1.0
    return names, cues, realised


def _run_method(method: str, seed: int, count: int, rho: float,
                height: int, width: int, channels: int) -> dict:
    dimension = height * width * channels
    key_dimension = height * width
    names_hsf, cues_hsf, realised_hsf = correlated_cues(seed, key_dimension, count, rho)
    names_vec, cues_vec, realised_vec = correlated_cues(seed, dimension, count, rho)
    if names_hsf != names_vec:
        raise AssertionError("method cue identities diverged")

    if method == "hsf":
        store = HolographicStateField(VOCABULARY, height=height, width=width,
                                      channels=channels, seed=seed)
        cues = cues_hsf
        realised = realised_hsf
        write = lambda name, value: store.write_pattern(cues[name], value)
        retrieve = lambda name: store.retrieve_pattern(cues[name]).value
    elif method == "hrr":
        store = HRRMemory(VOCABULARY, dimension=dimension, seed=seed)
        cues = cues_vec
        realised = realised_vec
        write = lambda name, value: store.write_vector(cues[name], value)
        retrieve = lambda name: store.retrieve_vector(cues[name]).value
    elif method == "hdc":
        store = HDCMemory(VOCABULARY, dimension=dimension, seed=seed)
        cues = cues_vec
        realised = realised_vec
        write = lambda name, value: store.write_vector(cues[name], value)
        retrieve = lambda name: store.retrieve_vector(cues[name]).value
    else:
        raise ValueError(method)

    records = [
        (name, VOCABULARY[((index // FAMILIES) * 3 + (index % FAMILIES) * 5 + seed)
                          % len(VOCABULARY)])
        for index, name in enumerate(names_hsf)
    ]
    for name, value in records:
        write(name, value)
    correct = sum(retrieve(name) == value for name, value in records)
    return {
        "accuracy": correct / count,
        "correct": correct,
        "numeric_payload_bytes": store.memory_bytes,
        "mean_realised_within_family_cosine": realised,
    }


def run_one(seed: int, count: int, rho: float, *,
            height: int, width: int, channels: int) -> dict:
    methods = {
        method: _run_method(method, seed, count, rho, height, width, channels)
        for method in ("hsf", "hrr", "hdc")
    }
    return {
        "seed": seed,
        "memories": count,
        "target_within_family_cosine": rho,
        "field_shape": [height, width, channels],
        "flattened_dimension": height * width * channels,
        "methods": methods,
    }


def run(seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS,
        similarities=DEFAULT_SIMILARITIES, *, height=16, width=16, channels=16):
    runs = [
        run_one(seed, count, rho, height=height, width=width, channels=channels)
        for rho in similarities for count in counts for seed in seeds
    ]
    aggregates = []
    for rho in similarities:
        for count in counts:
            group = [r for r in runs
                     if r["target_within_family_cosine"] == rho and r["memories"] == count]
            row = {"target_within_family_cosine": rho, "memories": count}
            for method in ("hsf", "hrr", "hdc"):
                values = [r["methods"][method]["accuracy"] for r in group]
                row[method] = {
                    "mean_accuracy": statistics.fmean(values),
                    "minimum_accuracy": min(values),
                    "mean_realised_within_family_cosine": statistics.fmean(
                        r["methods"][method]["mean_realised_within_family_cosine"]
                        for r in group),
                    "numeric_payload_bytes": group[0]["methods"][method]["numeric_payload_bytes"],
                }
            aggregates.append(row)
    return {
        "benchmark": "NSA correlated-key interference",
        "version": "1.0.0",
        "hypothesis": "Associative recall degrades as keys become correlated; representation structure changes the degradation curve.",
        "scientific_boundary": "Synthetic closed-vocabulary recall only. No semantic learning, biological fidelity, language task, or attention replacement is tested.",
        "key_generation": {
            "families": FAMILIES,
            "expected_within_family_cosine": "rho; independent bipolar prototypes with per-item coordinate flips",
            "flip_probability": "(1 - sqrt(rho)) / 2",
            "dimension_matching": "HSF spatial cue dimension is H*W; HRR/HDC cue dimension is H*W*C. Same family/item identities and target rho are used, but vectors are separately generated and realised cosine is reported for each.",
            "cross_family": "Independent family prototypes",
        },
        "configuration": {
            "seeds": list(seeds), "counts": list(counts),
            "target_similarities": list(similarities), "vocabulary": list(VOCABULARY),
            "height": height, "width": width, "channels": channels,
        },
        "aggregates": aggregates,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--similarities", nargs="+", type=float, default=list(DEFAULT_SIMILARITIES))
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--out", default="results/hsf_correlated_keys_v1.json")
    args = parser.parse_args()
    report = run(args.seeds, args.counts, args.similarities,
                 height=args.height, width=args.width, channels=args.channels)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["aggregates"]:
        print(f"rho={row['target_within_family_cosine']:.2f} n={row['memories']:>4} "
              f"HSF={row['hsf']['mean_accuracy']:.3f} "
              f"HRR={row['hrr']['mean_accuracy']:.3f} "
              f"HDC={row['hdc']['mean_accuracy']:.3f}")
    print(f"artifact={output}")


if __name__ == "__main__":
    main()
