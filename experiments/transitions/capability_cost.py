"""Benchmark transition-level cost of heterogeneous NSA state enforcement.

This experiment deliberately does not claim to measure neural-model capability.
It measures two quantities that must be established before a matched-model
capability study is meaningful:

* runtime overhead of the authoritative transition engine versus an
  unconstrained candidate assignment;
* candidate preservation / projection distortion under legal transition cones.

The benchmark uses deterministic synthetic heterogeneous states so the result
is reproducible and independent of model weights.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from random import Random
from typing import Any

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    CapabilityDomain,
    HeterogeneousState,
    NumericRangeDomain,
)
from nsa.core.transition_cone import TransitionCone, TransitionDirection
from nsa.transitions import TransitionEngine


@dataclass(frozen=True)
class BenchmarkSummary:
    seed: int
    samples: int
    dimensions: int
    unconstrained_ns_per_transition: float
    constrained_ns_per_transition: float
    overhead_ratio: float
    projected_fraction: float
    exact_legal_fraction: float
    mean_numeric_distortion: float
    max_numeric_distortion: float
    finite: bool


def _sample_state(rng: Random) -> tuple[HeterogeneousState[Any], TransitionCone[Any]]:
    domains = (
        NumericRangeDomain(0.0, 1.0),
        BooleanDomain(),
        CapabilityDomain(),
        NumericRangeDomain(0.0, 1.0),
    )
    source = HeterogeneousState(
        (
            rng.random(),
            bool(rng.randrange(2)),
            frozenset({"read"}) if rng.randrange(2) else frozenset(),
            rng.random(),
        ),
        domains,
    )
    candidate = HeterogeneousState(
        (
            rng.random(),
            bool(rng.randrange(2)),
            frozenset({"read", "write"}) if rng.randrange(2) else frozenset(),
            rng.random(),
        ),
        domains,
    )
    # Security/capability coordinates are intentionally restrictive; the two
    # numeric coordinates are used as continuous utility-like dimensions.
    cone = TransitionCone(
        (
            TransitionDirection.INCREASE,
            TransitionDirection.UNCHANGED,
            TransitionDirection.UNCHANGED,
            TransitionDirection.INCREASE,
        )
    )
    return source, candidate, cone


def run(seed: int = 42, samples: int = 2000) -> dict[str, Any]:
    rng = Random(seed)
    engine = TransitionEngine()
    cases = [_sample_state(rng) for _ in range(samples)]

    t0 = time.perf_counter_ns()
    unconstrained = [candidate for _, candidate, _ in cases]
    unconstrained_ns = time.perf_counter_ns() - t0

    projected = 0
    legal = 0
    distortions: list[float] = []
    t0 = time.perf_counter_ns()
    for source, candidate, cone in cases:
        result = engine.apply_heterogeneous(source, candidate, cone, project_illegal=True)
        legal += int(not result.projected)
        projected += int(result.projected)
        state = result.state
        # Only compare the continuous coordinates; discrete coordinates have
        # domain-specific semantics and no meaningful Euclidean distortion.
        distortion = abs(state.values[0] - candidate.values[0]) + abs(
            state.values[3] - candidate.values[3]
        )
        distortions.append(distortion)
    constrained_ns = time.perf_counter_ns() - t0

    unconstrained_per = unconstrained_ns / samples
    constrained_per = constrained_ns / samples
    overhead = constrained_per / unconstrained_per if unconstrained_per else float("inf")
    mean_distortion = sum(distortions) / len(distortions)
    max_distortion = max(distortions, default=0.0)

    finite = all(
        value == value and value not in (float("inf"), float("-inf"))
        for value in (
            unconstrained_per,
            constrained_per,
            overhead,
            mean_distortion,
            max_distortion,
        )
    )

    return asdict(
        BenchmarkSummary(
            seed=seed,
            samples=samples,
            dimensions=4,
            unconstrained_ns_per_transition=unconstrained_per,
            constrained_ns_per_transition=constrained_per,
            overhead_ratio=overhead,
            projected_fraction=projected / samples,
            exact_legal_fraction=legal / samples,
            mean_numeric_distortion=mean_distortion,
            max_numeric_distortion=max_distortion,
            finite=finite,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.samples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
