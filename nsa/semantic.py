"""Semantic classification boundary.

The reference implementation is intentionally conservative and deterministic.
It is an adapter boundary, not a claim that keywords constitute semantic
understanding. A trained classifier can implement the same protocol later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .normative import NormativeAssessment, NormativeState


@dataclass(frozen=True)
class SemanticAssessment:
    categories: Sequence[str]
    confidence: float
    normative: NormativeAssessment


class SemanticClassifier(Protocol):
    def classify(self, text: str) -> SemanticAssessment:
        ...


class ReferenceSemanticClassifier:
    """Small reference adapter built on deterministic policy patterns."""

    def __init__(self, rules: Sequence[tuple[str, Sequence[str]]]):
        self._rules = tuple((name, tuple(patterns)) for name, patterns in rules)

    def classify(self, text: str) -> SemanticAssessment:
        lowered = text.lower()
        categories = [
            name for name, patterns in self._rules
            if any(pattern.lower() in lowered for pattern in patterns)
        ]
        harmful = 1.0 if any("harm" in c or "weapon" in c for c in categories) else 0.0
        sensitive = 1.0 if any("sensitive" in c or "protected" in c for c in categories) else 0.0
        confidence = 0.95 if categories else 0.90
        normative = NormativeAssessment(
            NormativeState(
                {"harm": harmful, "sensitivity": sensitive},
                confidence,
                source="reference-semantic-classifier",
            ),
            rationale="deterministic reference classification",
        )
        return SemanticAssessment(categories, confidence, normative)
