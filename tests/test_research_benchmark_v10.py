import pytest

np = pytest.importorskip("numpy")

from experiments.research_v10 import gated_memory_benchmark as v10


def test_stream_has_reminders_with_typos_genuine_updates_and_an_untrusted_update():
    events, truth, typos = v10.make_stream(3, "train")
    assert typos and len(truth) == 20
    assert any(s == v10.UNTRUSTED_SOURCE for _, s in events)
    assert all(v not in typos for v in truth.values())


def test_ungated_rules_extractor_is_corrupted_by_typos_and_the_gate_reduces_work_and_corruption():
    ungated = [v10.run_one("rules", None, "train", s) for s in range(4)]
    gated = [v10.run_one("rules", 0.75, "train", s) for s in range(4)]
    assert sum(r.corrupted for r in ungated) > 0 and sum(r.corrupted for r in gated) < sum(r.corrupted for r in ungated)
    assert sum(r.lines_extracted for r in gated) < 0.8 * sum(r.lines_extracted for r in ungated)


def test_run_validates_and_unknown_extractor_is_rejected():
    with pytest.raises(ValueError):
        v10.run(seeds=())
    with pytest.raises(ValueError):
        v10.run_one("nope", None, "train", 0)
