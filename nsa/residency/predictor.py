"""Predict which neural regions will be needed next."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence
from nsa.residency.types import NeuralRegion

class ResidencyPredictor:
    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        raise NotImplementedError

@dataclass
class HeuristicResidencyPredictor(ResidencyPredictor):
    transition_counts: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    tag_affinity: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    def observe_transition(self, previous: str | None, current: str) -> None:
        if previous is None: return
        row = self.transition_counts.setdefault(previous, {})
        row[current] = row.get(current, 0.0) + 1.0
    def observe_state_tags(self, tags: Iterable[str], region_id: str) -> None:
        for tag in tags:
            row = self.tag_affinity.setdefault(tag, {})
            row[region_id] = row.get(region_id, 0.0) + 1.0
    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        return {k: (v / total if total > 0 else 0.0) for k,v in values.items()}
    def predict(self, regions: Sequence[NeuralRegion], state: Mapping[str, object], current_region: str | None = None) -> dict[str, float]:
        scores = {r.region_id: 0.0 for r in regions}
        if current_region in self.transition_counts:
            for rid,value in self._normalize(self.transition_counts[current_region]).items():
                if rid in scores: scores[rid] += 0.65 * value
        raw_tags = state.get("tags", ())
        tags = raw_tags if isinstance(raw_tags, (list, tuple, set, frozenset)) else ()
        for tag in tags:
            for rid,value in self._normalize(self.tag_affinity.get(str(tag), {})).items():
                if rid in scores: scores[rid] += 0.35 * value
        return {rid: min(1.0, score) for rid,score in scores.items()}
