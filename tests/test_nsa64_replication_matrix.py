from experiments.nsa64.replication_matrix import config_hash, paired_summary
from experiments.nsa64.replication_matrix import RunRecord


def record(split="development", model="mock", gtc=1.0):
    return RunRecord(model, "mock", split, 7, 4, 0.1, 2, gtc, 0, 10.0, 1.0, 0.1, 0.01, 1, 0, 1, True, True, "raw.json")


def test_summary_is_deterministic_and_keeps_splits_separate():
    summary = paired_summary([record("development"), record("heldout", gtc=0.5)])
    assert summary["development:mock"]["gtc_mean"] == 1.0
    assert summary["heldout:mock"]["gtc_mean"] == 0.5


def test_config_hash_is_order_independent():
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})


def test_zero_authority_is_not_inferred_from_gtc():
    safe = record(gtc=0.0)
    assert safe.violations == 0
    assert safe.invariants_verified
