"""Pluggable Neural and Reference Semantic Classifiers for NSA Normative State (Phase F).

Formulates the assessment mapping:
    f_theta(x, c, m) -> (nu, u_hat, p)
where:
    x     : Input text or prompt
    c     : Continuous cognitive state context
    m     : Episodic working memory context
    nu    : Output NormativeState
    u_hat : Uncertainty estimate in [0, 1]
    p     : Provenance metadata with SHA-256 assessment ID
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from nsa.normative.state import (
    ConfidenceCalibrator,
    NormativeAssessmentMetadata,
    NormativeClass,
    NormativeState,
)


class SemanticClassifierProtocol(Protocol):
    """Formal protocol for semantic and normative classification."""

    def classify_normative(
        self,
        text: str,
        cognitive_context: Optional[Mapping[str, float]] = None,
        memory_context: Optional[Mapping[str, float]] = None,
    ) -> NormativeState:
        """Evaluate input text within cognitive/memory context and produce typed NormativeState."""
        ...


class ReferenceSemanticClassifier:
    """Deterministic reference classifier using calibrated pattern matching."""

    def __init__(
        self,
        rules: Optional[Mapping[str, Sequence[str]]] = None,
        version: str = "ref-1.0",
        base_confidence: float = 0.90,
    ) -> None:
        self.version = version
        self.base_confidence = max(0.1, min(1.0, float(base_confidence)))
        self.rules = rules or {
            "harm": ["harm", "attack", "exploit", "destroy", "kill", "wipe", "format", "rm -rf"],
            "sensitivity": ["password", "secret", "private", "credential", "token", "ssn", "confidential"],
            "deception": ["lie", "mislead", "fabricate", "falsify"],
        }
        self._seq = 0

    def classify_normative(
        self,
        text: str,
        cognitive_context: Optional[Mapping[str, float]] = None,
        memory_context: Optional[Mapping[str, float]] = None,
    ) -> NormativeState:
        self._seq += 1
        lowered = text.lower()
        values = {}

        for category, patterns in self.rules.items():
            matched_count = sum(1 for p in patterns if p in lowered)
            score = min(1.0, matched_count * 0.4)
            values[category] = score

        # Cognitive modulation
        if cognitive_context:
            for k, mod in cognitive_context.items():
                if k in values:
                    values[k] = max(0.0, min(1.0, values[k] + mod * 0.1))

        meta = NormativeAssessmentMetadata.create(
            source="reference_semantic_classifier",
            classifier_version=self.version,
            sequence_id=self._seq,
            policy_version="default-policy",
            confidence=self.base_confidence,
            values_digest=NormativeState(values, self.base_confidence).digest(),
        )

        return NormativeState(
            values=values,
            confidence=self.base_confidence,
            source="reference_semantic_classifier",
            metadata=meta,
        )

    def classify(self, text: str) -> Sequence[str]:
        """Conforms to nsa.enforcement.PolicyClassifier interface."""
        nu = self.classify_normative(text)
        categories = []
        for cat, score in nu.values.items():
            if score >= 0.4:
                categories.append(cat)
        return tuple(categories)


class CalibratedNeuralClassifier:
    """Wrapper that applies temperature scaling and calibration to any underlying classifier."""

    def __init__(
        self,
        base_classifier: SemanticClassifierProtocol,
        temperature: float = 1.0,
    ) -> None:
        self.base = base_classifier
        self.temperature = max(0.1, float(temperature))

    def classify_normative(
        self,
        text: str,
        cognitive_context: Optional[Mapping[str, float]] = None,
        memory_context: Optional[Mapping[str, float]] = None,
    ) -> NormativeState:
        raw_state = self.base.classify_normative(text, cognitive_context, memory_context)
        calibrated_conf = ConfidenceCalibrator.apply_temperature_scaling(
            raw_state.confidence, temperature=self.temperature
        )
        return raw_state.with_confidence(calibrated_conf)


__all__ = [
    "SemanticClassifierProtocol",
    "ReferenceSemanticClassifier",
    "CalibratedNeuralClassifier",
]
