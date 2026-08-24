"""Explicit normative/value state for the NSA control plane.

This module deliberately does not claim to encode morality. It provides a
small, typed substrate for representing policy-relevant normative assessments
with explicit uncertainty, so a future trained semantic/normative model can be
plugged in without changing the policy API.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class NormativeClass(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormativeState:
    """A bounded normative assessment, separate from security authority."""

    values: Mapping[str, float]
    confidence: float
    source: str = "reference"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        for key, value in self.values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"normative value {key!r} must be in [0, 1]")

    @property
    def dominant(self) -> NormativeClass:
        if not self.values or self.confidence < 0.5:
            return NormativeClass.UNKNOWN
        harmful = float(self.values.get("harm", 0.0))
        sensitive = float(self.values.get("sensitivity", 0.0))
        if harmful >= 0.75:
            return NormativeClass.HARMFUL
        if sensitive >= 0.5:
            return NormativeClass.SENSITIVE
        return NormativeClass.SAFE

    def with_confidence(self, confidence: float) -> "NormativeState":
        return NormativeState(dict(self.values), confidence, self.source)


@dataclass(frozen=True)
class NormativeAssessment:
    """Result of semantic/normative evaluation before authority enforcement."""

    state: NormativeState
    rationale: str = ""

    @property
    def uncertain(self) -> bool:
        return self.state.confidence < 0.75 or self.state.dominant is NormativeClass.UNKNOWN
