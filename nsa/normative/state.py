"""Typed normative/value state and immutable provenance metadata for the NSA control plane."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class NormativeClass(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormativeAssessmentMetadata:
    """Immutable provenance and audit metadata attached to a normative assessment."""

    assessment_id: str
    source: str
    classifier_version: str
    timestamp_utc: float
    sequence_id: int
    policy_version: str
    confidence: float
    parent_event_id: Optional[str] = None
    calibration_score: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.calibration_score <= 1.0:
            raise ValueError("calibration_score must be in [0, 1]")

    @classmethod
    def create(
        cls,
        source: str,
        classifier_version: str,
        sequence_id: int,
        policy_version: str,
        confidence: float,
        values_digest: str,
        parent_event_id: Optional[str] = None,
        calibration_score: float = 1.0,
        timestamp_utc: Optional[float] = None,
    ) -> "NormativeAssessmentMetadata":
        ts = timestamp_utc if timestamp_utc is not None else time.time()
        raw_seed = f"{source}:{classifier_version}:{sequence_id}:{policy_version}:{confidence:.6f}:{values_digest}:{parent_event_id}:{ts:.6f}"
        aid = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest()[:16]
        return cls(
            assessment_id=aid,
            source=source,
            classifier_version=classifier_version,
            timestamp_utc=ts,
            sequence_id=sequence_id,
            policy_version=policy_version,
            confidence=float(confidence),
            parent_event_id=parent_event_id,
            calibration_score=float(calibration_score),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormativeState:
    """Bounded normative assessment state, separate from security authority."""

    values: Mapping[str, float]
    confidence: float
    source: str = "reference"
    metadata: Optional[NormativeAssessmentMetadata] = None

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
        return NormativeState(dict(self.values), confidence, self.source, self.metadata)

    def digest(self) -> str:
        """Deterministic cryptographic fingerprint of normative state values."""
        ordered = sorted((k, round(float(v), 6)) for k, v in self.values.items())
        serialized = json.dumps({"vals": ordered, "conf": round(self.confidence, 6)}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": dict(self.values),
            "confidence": self.confidence,
            "source": self.source,
            "digest": self.digest(),
            "metadata": self.metadata.to_dict() if self.metadata is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormativeState":
        meta_raw = data.get("metadata")
        meta = NormativeAssessmentMetadata(**meta_raw) if meta_raw is not None else None
        return cls(
            values=dict(data.get("values", {})),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source", "reference")),
            metadata=meta,
        )


class ConfidenceCalibrator:
    """Confidence calibration utilities for mapping model uncertainty into calibrated risk bounds."""

    @staticmethod
    def calibrate_brier(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
        """Compute Brier Score: (1/N) * sum((p_i - o_i)^2). Lower is better (0.0 = perfect)."""
        if len(predictions) != len(outcomes) or len(predictions) == 0:
            raise ValueError("predictions and outcomes must have equal non-zero length")
        return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)

    @staticmethod
    def apply_temperature_scaling(raw_confidence: float, temperature: float = 1.0) -> float:
        """Smooth or sharpen raw model confidence using temperature scaling."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        # Scaled logit projection
        p = max(1e-6, min(1.0 - 1e-6, raw_confidence))
        import math
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return max(0.0, min(1.0, calibrated))


@dataclass(frozen=True)
class NormativeAssessment:
    """Result of semantic/normative evaluation before authority enforcement."""

    state: NormativeState
    rationale: str = ""

    @property
    def uncertain(self) -> bool:
        return self.state.confidence < 0.75 or self.state.dominant is NormativeClass.UNKNOWN


__all__ = [
    "NormativeClass",
    "NormativeAssessmentMetadata",
    "NormativeState",
    "ConfidenceCalibrator",
    "NormativeAssessment",
]
