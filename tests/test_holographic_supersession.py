import numpy as np
import pytest

from experiments.holographic_state_field.field import HolographicStateField
from experiments.holographic_state_field.supersession_benchmark import METHODS, run, run_one

VOCAB = tuple(f"VALUE_{i}" for i in range(8))


def _field(**kw):
    return HolographicStateField(VOCAB, height=16, width=16, channels=16, seed=7, **kw)


def test_delta_write_supersedes_in_place_where_additive_write_does_not():
    additive, delta = _field(), _field()
    for f, write in ((additive, additive.write), (delta, delta.write_delta)):
        write("k", "VALUE_1")
        write("k", "VALUE_2")
    assert delta.retrieve("k").value == "VALUE_2"
    assert delta.retrieve("k").scores["VALUE_2"] == pytest.approx(1.0, abs=1e-4)
    # additive superposition keeps both traces: the old value scores as high as the new one
    scores = additive.retrieve("k").scores
    assert scores["VALUE_1"] == pytest.approx(scores["VALUE_2"], abs=1e-4)


def test_delta_write_is_idempotent_and_returns_the_residual():
    f = _field()
    assert f.write_delta("k", "VALUE_3") > 0.9
    before = f.field_energy
    assert f.write_delta("k", "VALUE_3") == pytest.approx(0.0, abs=1e-4)
    assert f.field_energy == pytest.approx(before, rel=1e-5)


def test_delta_write_leaves_orthogonal_keys_untouched():
    f = _field()
    a = np.zeros(256, dtype=np.float32); a[0] = 1.0
    b = np.zeros(256, dtype=np.float32); b[1] = 1.0
    f.write_pattern_delta(a, "VALUE_1")
    f.write_pattern_delta(b, "VALUE_2")
    f.write_pattern_delta(b, "VALUE_5")
    assert f.retrieve_pattern(a).value == "VALUE_1" and f.retrieve_pattern(b).value == "VALUE_5"
    assert f.retrieve_pattern(a).scores["VALUE_1"] == pytest.approx(1.0, abs=1e-5)


def test_delta_write_validates_inputs():
    f = _field()
    with pytest.raises(KeyError):
        f.write_delta("k", "NOT_IN_VOCAB")
    with pytest.raises(ValueError):
        f.write_delta("k", "VALUE_1", rate=0.0)
    with pytest.raises(ValueError):
        f.write_pattern_delta(np.zeros(256, dtype=np.float32), "VALUE_1")


def test_benchmark_smoke_and_delta_beats_additive_on_updates():
    rows = {m: run_one(m, 7, 32, 0.0, 0.5, 16, 16, 16) for m in METHODS}
    assert rows["hsf_delta"]["updated_accuracy"] > rows["hsf_additive"]["updated_accuracy"]
    assert rows["hsf_delta"]["stale_rate"] < rows["hsf_additive"]["stale_rate"]
    out = run(seeds=(7,), counts=(16,), similarities=(0.0,), methods=("hsf_delta", "hdc_cleanup"))
    assert len(out["runs"]) == 2 and out["aggregates"][0]["method"] == "hsf_delta"


def test_benchmark_rejects_bad_configuration():
    with pytest.raises(ValueError):
        run(seeds=(), counts=(16,), similarities=(0.0,))
    with pytest.raises(ValueError):
        run(update_fraction=0.0)


def test_cleanup_threshold_is_calibrated_per_method_so_hdc_cleanup_actually_fires():
    add = run_one("hdc_additive", 7, 32, 0.0, 0.5, 16, 16, 16)
    clean = run_one("hdc_cleanup", 7, 32, 0.0, 0.5, 16, 16, 16)
    assert clean["updated_accuracy"] > add["updated_accuracy"]
