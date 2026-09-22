"""Predict which neural regions will be needed next."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
from nsa.residency.types import NeuralRegion


def state_tags(state: Mapping[str, object] | None) -> tuple[str, ...]:
    """Extract residency tags from a state mapping without trusting its shape."""
    if not isinstance(state, Mapping):
        return ()
    raw = state.get("tags", ())
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple(str(tag) for tag in raw)
    return ()


def normalize_counts(values: Mapping[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total > 0 else {}


def blend_evidence(
    region_ids: Iterable[str],
    transition_row: Mapping[str, float],
    tag_rows: Sequence[Mapping[str, float]],
    transition_weight: float,
    tag_weight: float,
) -> dict[str, float]:
    """Mix the evidence sources that actually exist into per-region probabilities.

    Weights are renormalised over the sources that are present, so a fully
    deterministic transition history yields probability 1.0 rather than being
    capped at its nominal weight. Missing evidence never dilutes present evidence.
    """
    scores = {rid: 0.0 for rid in region_ids}
    sources: list[tuple[float, dict[str, float]]] = []
    transitions = normalize_counts(transition_row)
    if transitions:
        sources.append((transition_weight, transitions))
    tag_distributions = [dist for dist in (normalize_counts(row) for row in tag_rows) if dist]
    if tag_distributions:
        merged: dict[str, float] = {}
        for dist in tag_distributions:
            for rid, value in dist.items():
                merged[rid] = merged.get(rid, 0.0) + value / len(tag_distributions)
        sources.append((tag_weight, merged))
    total_weight = sum(weight for weight, _ in sources)
    if total_weight <= 0:
        return scores
    for weight, distribution in sources:
        for rid, value in distribution.items():
            if rid in scores:
                scores[rid] += weight * value / total_weight
    return {rid: min(1.0, score) for rid, score in scores.items()}


class ResidencyPredictor:
    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class HeuristicResidencyPredictor(ResidencyPredictor):
    transition_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    tag_affinity: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    transition_weight: float = 0.65
    tag_weight: float = 0.35

    def observe_transition(self, previous: str | None, current: str) -> None:
        if previous is None:
            return
        row = self.transition_counts.setdefault(previous, {})
        row[current] = row.get(current, 0.0) + 1.0

    def observe_state_tags(self, tags: Iterable[str], region_id: str) -> None:
        for tag in tags:
            row = self.tag_affinity.setdefault(str(tag), {})
            row[region_id] = row.get(region_id, 0.0) + 1.0

    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        transition_row = self.transition_counts.get(current_region, {}) if current_region is not None else {}
        tag_rows = [self.tag_affinity[tag] for tag in state_tags(state) if tag in self.tag_affinity]
        return blend_evidence(
            (r.region_id for r in regions), transition_row, tag_rows, self.transition_weight, self.tag_weight
        )
