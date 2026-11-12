import numpy as np
import pytest

from experiments.organic.dog_test import Field, Knn, Organic, Separator, World, cue_end_key, run, run_one, train_events


def test_trailing_field_still_holds_the_cue_when_the_delayed_outcome_arrives():
    w = World.make(3)
    f = Field()
    for _ in range(4):
        f.step(w.prototypes[0])
    at_cue_end = f.key()
    for _ in range(3):
        f.step(np.zeros(24, dtype=np.float32))
    assert float(at_cue_end @ f.key()) > 0.95          # decays, direction preserved: that is what lets the outcome bind to it


def test_separator_is_sparse_and_unit_norm():
    code = Separator(1)(np.ones(24, dtype=np.float32) / np.sqrt(24))
    assert np.count_nonzero(code) == 51 and np.linalg.norm(code) == pytest.approx(1.0, abs=1e-5)


def test_organic_agent_learns_rewarding_cue_with_a_fixed_size_memory():
    w = World.make(5)
    a = Organic(5)
    assert a.predict(cue_end_key(w, 0, 0.3)) == 0.0
    train_events(a, w, 200)
    assert a.predict(cue_end_key(w, 0, 0.3)) > 0.4 and a.predict(cue_end_key(w, 4, 0.3)) < -0.4
    assert a.M.shape == (1024,)


def test_surprise_gate_skips_writes_for_predictable_events():
    w = World.make(5)
    a = Organic(5)
    key = cue_end_key(w, 6, 0.3)                                    # a decoy: outcome 0, prediction 0 -> no surprise
    a.observe(key, 0.0, a.predict(key))
    assert not a.M.any() and a.buffer == []


def test_knn_respects_its_float_budget_and_policies_differ():
    k = Knn("fifo", 25 * 4, 1)
    for i in range(10):
        k.observe(np.eye(24, dtype=np.float32)[i], float(i), 0.0)
    assert len(k.keys) == 4 and k.out == [6.0, 7.0, 8.0, 9.0]


def test_run_one_reports_curve_and_retention_and_run_validates():
    r = run_one("organic", 1, events=(25, 50), distractor_events=20)
    assert [c["events"] for c in r["curve"]] == [25, 50] and "after_distractors" in r
    with pytest.raises(ValueError):
        run(seeds=())
