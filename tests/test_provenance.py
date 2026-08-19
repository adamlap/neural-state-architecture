import pytest

from nsa.provenance import Evidence, ProvenanceRecord, ProvenanceStore


def test_provenance_is_append_only():
    store = ProvenanceStore()
    first = ProvenanceRecord("claim-1", "observation", producer="sensor")
    second = store.append(first)
    assert store.records == ()
    assert second.records == (first,)


def test_duplicate_claim_is_rejected():
    record = ProvenanceRecord("claim-1", "observation")
    store = ProvenanceStore().append(record)
    with pytest.raises(ValueError):
        store.append(record)


def test_evidence_reliability_is_bounded():
    with pytest.raises(ValueError):
        Evidence("e1", "source", "measurement", reliability=1.1)


def test_evidence_can_be_added_without_mutating_record():
    record = ProvenanceRecord("claim-1", "derived")
    evidence = Evidence("e1", "sensor", "measurement", reliability=0.9)
    enriched = record.with_evidence(evidence)
    assert record.evidence == ()
    assert enriched.evidence == (evidence,)
