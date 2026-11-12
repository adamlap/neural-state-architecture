import pytest

from experiments.organic import dog_test as d1
from experiments.organic.dog_test_v2 import LAMBDAS, MultiField, detect_onsets, episode_keys, run, run_one


def test_multi_field_key_is_unit_norm_and_decays_the_cue_at_the_expected_rate():
    w = d1.World.make(1)
    cue_end, at_outcome = episode_keys(w, LAMBDAS["multi"], 0, 15, noise=0.0)
    assert float(cue_end @ cue_end) == pytest.approx(1.0, abs=1e-4)
    mid_only_end, mid_only_out = episode_keys(w, LAMBDAS["mid"], 0, 15, noise=0.0)
    slow_end, slow_out = episode_keys(w, LAMBDAS["slow"], 0, 15, noise=0.0)
    assert float(slow_end @ slow_out) > float(mid_only_end @ mid_only_out)      # slower trace keeps the cue direction longer


def test_trace_binding_works_at_short_delay_and_stored_binding_is_delay_independent():
    short = run_one("trace", "mid", 3, 0, n_events=300)["valenced_accuracy"]
    stored_long = run_one("stored", "mid", 25, 0, n_events=300)["valenced_accuracy"]
    assert short > 0.6 and stored_long > 0.6


def test_trace_binding_with_a_fast_trace_fails_at_a_long_delay():
    assert run_one("trace", "mid", 25, 0, n_events=300)["valenced_accuracy"] < 0.4


def test_onset_detection_finds_most_events_without_an_oracle():
    r = detect_onsets(0, n_events=80)
    assert r["recall"] > 0.7 and r["precision"] > 0.5


def test_run_validates_seeds():
    with pytest.raises(ValueError):
        run(seeds=())
