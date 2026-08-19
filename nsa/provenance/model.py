"""Immutable provenance records for claims and state observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: str
    kind: str
    reliability: float = 0.5
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")


@dataclass(frozen=True)
class ProvenanceRecord:
    claim_id: str
    claim_type: str
    parent_claims: Tuple[str, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    producer: str = "unknown"

    def with_evidence(self, evidence: Evidence) -> "ProvenanceRecord":
        return ProvenanceRecord(
            claim_id=self.claim_id,
            claim_type=self.claim_type,
            parent_claims=self.parent_claims,
            evidence=self.evidence + (evidence,),
            producer=self.producer,
        )


@dataclass(frozen=True)
class ProvenanceStore:
    records: Tuple[ProvenanceRecord, ...] = ()

    def append(self, record: ProvenanceRecord) -> "ProvenanceStore":
        if any(r.claim_id == record.claim_id for r in self.records):
            raise ValueError(f"duplicate claim_id: {record.claim_id}")
        return ProvenanceStore(self.records + (record,))

    def get(self, claim_id: str) -> ProvenanceRecord:
        for record in self.records:
            if record.claim_id == claim_id:
                return record
        raise KeyError(claim_id)


__all__ = ["Evidence", "ProvenanceRecord", "ProvenanceStore"]
