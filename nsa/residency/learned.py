"""Online residency predictor trained from execution traces.

This is a count-based (Markov transition + state-tag co-occurrence) model that
updates as regions execute. It is learned in the sense that it is fitted online
from execution telemetry; it is not a neural model.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from nsa.residency.predictor import ResidencyPredictor, blend_evidence, normalize_counts, state_tags
from nsa.residency.types import NeuralRegion

@dataclass
class OnlineResidencyPredictor(ResidencyPredictor):
    transition_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    state_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    observations: int = 0
    transition_weight: float = 0.75
    tag_weight: float = 0.25

    def observe(self, previous: str | None, current: str, state: Mapping[str, object] | None = None) -> None:
        self.observations += 1
        if previous is not None:
            row = self.transition_counts.setdefault(previous, {})
            row[current] = row.get(current, 0.0) + 1.0
        for tag in state_tags(state):
            row = self.state_counts.setdefault(tag, {})
            row[current] = row.get(current, 0.0) + 1.0

    @staticmethod
    def _norm(row: Mapping[str, float]) -> dict[str, float]:
        return normalize_counts(row)

    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        transition_row = self.transition_counts.get(current_region, {}) if current_region is not None else {}
        tag_rows = [self.state_counts[tag] for tag in state_tags(state) if tag in self.state_counts]
        return blend_evidence(
            (r.region_id for r in regions), transition_row, tag_rows, self.transition_weight, self.tag_weight
        )
