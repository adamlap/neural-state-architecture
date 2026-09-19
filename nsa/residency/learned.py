"""Learned residency predictor trained from execution traces."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from nsa.residency.predictor import ResidencyPredictor
from nsa.residency.types import NeuralRegion

@dataclass
class OnlineResidencyPredictor(ResidencyPredictor):
    transition_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    state_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    observations: int = 0

    def observe(self, previous: str | None, current: str, state: Mapping[str, object] | None = None) -> None:
        self.observations += 1
        if previous is not None:
            row = self.transition_counts.setdefault(previous, {})
            row[current] = row.get(current, 0.0) + 1.0
        for tag in (state or {}).get("tags", ()):
            row = self.state_counts.setdefault(str(tag), {})
            row[current] = row.get(current, 0.0) + 1.0

    @staticmethod
    def _norm(row: Mapping[str, float]) -> dict[str, float]:
        total = sum(row.values())
        return {k: v / total for k,v in row.items()} if total else {}

    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        scores = {r.region_id: 0.0 for r in regions}
        if current_region in self.transition_counts:
            for rid,p in self._norm(self.transition_counts[current_region]).items():
                if rid in scores: scores[rid] += 0.75*p
        for tag in state.get("tags", ()):
            for rid,p in self._norm(self.state_counts.get(str(tag), {})).items():
                if rid in scores: scores[rid] += 0.25*p
        return {rid: min(1.0, value) for rid,value in scores.items()}
