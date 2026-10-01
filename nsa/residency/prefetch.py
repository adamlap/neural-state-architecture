"""Adaptive next-region prediction for any ordered execution backend."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefetchDecision:
    region: str
    confidence: float


class SequentialPrefetcher:
    """Learn transition probabilities from observed region execution.

    The predictor starts with a deterministic sequential prior and adapts when
    the runtime observes a different transition.  It emits only logical
    decisions; a backend decides how and where data is actually prefetched.
    """

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def observe(self, previous: str, current: str) -> None:
        self._counts[previous][current] += 1

    def predict(self, previous: str, candidates: list[str], top_k: int = 1) -> list[PrefetchDecision]:
        if top_k <= 0 or not candidates:
            return []

        counts = self._counts.get(previous, {})
        total = sum(counts.get(candidate, 0) for candidate in candidates)
        if total:
            ranked = sorted(
                candidates,
                key=lambda candidate: counts.get(candidate, 0),
                reverse=True,
            )
            return [
                PrefetchDecision(candidate, counts.get(candidate, 0) / total)
                for candidate in ranked[:top_k]
                if counts.get(candidate, 0)
            ]

        # Cold start: ordered execution is the strongest hardware-independent prior.
        return [
            PrefetchDecision(candidate, 1.0 / len(candidates))
            for candidate in candidates[:top_k]
        ]
