"""Held-out benchmark for competitive Hebbian self-organisation.

Family labels are used only to score cluster purity and map clusters to
families after training. The learner itself receives unlabelled vectors.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .plasticity import CompetitiveHebbianMap

FAMILIES = 4


def make_family_data(*, seed: int, dimension: int, train_per_family: int,
                     test_per_family: int, rho: float):
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if dimension < 2 or min(train_per_family, test_per_family) < 1:
        raise ValueError("dimension >= 2 and sample counts >= 1 are required")
    rng = np.random.default_rng(seed)
    prototypes = rng.choice(np.array([-1.0, 1.0], dtype=np.float32),
                             size=(FAMILIES, dimension))
    flip_probability = (1.0 - np.sqrt(rho)) / 2.0

    def sample(per_family):
        vectors, labels = [], []
        for family in range(FAMILIES):
            for _ in range(per_family):
                cue = prototypes[family].copy()
                cue[rng.random(dimension) < flip_probability] *= -1.0
                vectors.append(cue / np.sqrt(np.float32(dimension)))
                labels.append(family)
        order = rng.permutation(len(labels))
        return np.asarray(vectors, dtype=np.float32)[order], np.asarray(labels)[order]

    train_x, train_y = sample(train_per_family)
    test_x, test_y = sample(test_per_family)
    return train_x, train_y, test_x, test_y


def _cluster_mapping(clusters, labels, units):
    mapping = {}
    for unit in range(units):
        members = labels[clusters == unit]
        if len(members):
            values, counts = np.unique(members, return_counts=True)
            mapping[unit] = int(values[np.argmax(counts)])
    return mapping


def _mapped_accuracy(clusters, labels, mapping):
    if not mapping:
        return 0.0
    predictions = np.asarray([mapping.get(int(c), -1) for c in clusters])
    return float(np.mean(predictions == labels))


def run(*, seeds=(7, 17, 37, 73, 137), similarities=(0.5, 0.75, 0.9),
        dimension=512, train_per_family=128, test_per_family=64,
        units=4, epochs=8, learning_rate=0.2, neighbourhood=0.5):
    rows = []
    for rho in similarities:
        for seed in seeds:
            train_x, train_y, test_x, test_y = make_family_data(
                seed=seed, dimension=dimension, train_per_family=train_per_family,
                test_per_family=test_per_family, rho=rho)
            learner = CompetitiveHebbianMap(
                units=units, dimension=dimension, seed=seed + 1000,
                learning_rate=learning_rate, neighbourhood=neighbourhood)
            initial = learner.weights
            initial_train = learner.predict(train_x)
            initial_map = _cluster_mapping(initial_train, train_y, units)
            initial_test = _mapped_accuracy(learner.predict(test_x), test_y, initial_map)

            learner.fit(train_x, epochs=epochs, seed=seed + 2000)
            train_clusters = learner.predict(train_x)
            mapping = _cluster_mapping(train_clusters, train_y, units)
            test_clusters = learner.predict(test_x)
            rows.append({
                "seed": seed,
                "rho": rho,
                "fixed_random_test_accuracy": initial_test,
                "hebbian_test_accuracy": _mapped_accuracy(test_clusters, test_y, mapping),
                "hebbian_train_purity": _mapped_accuracy(train_clusters, train_y, mapping),
                "occupied_units": len(set(int(v) for v in test_clusters)),
                "updates": learner.updates,
                "prototype_payload_bytes": learner.memory_bytes,
                "weight_change_l2": float(np.linalg.norm(learner.weights - initial)),
            })
    aggregates = []
    for rho in similarities:
        group = [row for row in rows if row["rho"] == rho]
        aggregates.append({
            "rho": rho,
            "fixed_random_mean_test_accuracy": statistics.fmean(
                row["fixed_random_test_accuracy"] for row in group),
            "hebbian_mean_test_accuracy": statistics.fmean(
                row["hebbian_test_accuracy"] for row in group),
            "hebbian_minimum_test_accuracy": min(
                row["hebbian_test_accuracy"] for row in group),
            "mean_occupied_units": statistics.fmean(
                row["occupied_units"] for row in group),
        })
    return {
        "benchmark": "NSA competitive Hebbian self-organisation",
        "version": "1.0.0",
        "scientific_boundary": (
            "Synthetic family clustering only. No labels enter learning; labels "
            "are used after training to score cluster-to-family correspondence. "
            "This is not semantic learning or evidence of biological equivalence."
        ),
        "config": {
            "seeds": list(seeds), "similarities": list(similarities),
            "dimension": dimension, "train_per_family": train_per_family,
            "test_per_family": test_per_family, "units": units, "epochs": epochs,
            "learning_rate": learning_rate, "neighbourhood": neighbourhood,
        },
        "runs": rows,
        "aggregates": aggregates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(result["runs"]), "aggregates": result["aggregates"],
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
