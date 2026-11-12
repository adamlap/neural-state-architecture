import pytest

from experiments.organic import dog_test_v3 as v3


def test_mixed_world_assigns_different_delays_to_valenced_cues():
    w = v3.MixedWorld.make(1)
    assert set(w.delays.values()) == set(v3.DELAYS)


def test_no_single_timescale_covers_a_mixed_delay_world():
    fast = v3.run_one("single_0.8", 0, n_events=400)
    assert fast["by_delay"]["3"] > 0.8 and fast["by_delay"]["80"] < 0.5


def test_joint_codes_cover_more_delays_than_a_fast_single_trace():
    joint = sum(v3.run_one("joint_codes", s, n_events=400)["valenced"] for s in range(3)) / 3
    fast = sum(v3.run_one("single_0.8", s, n_events=400)["valenced"] for s in range(3)) / 3
    assert joint > fast + 0.1


def test_detectors_find_strong_cues_and_the_sweep_runs():
    xs, truth = v3._stream(0, 1.0, 0.2, 40)
    r, p = v3._score(v3.detect_predictive(xs, 4.0), truth)
    assert r > 0.8 and p > 0.8
    assert len(v3.detection_sweep(seeds=(0,), scales=(1.0,), noises=(0.2,), n_events=30)["rows"]) == 2


def test_run_validates_seeds():
    with pytest.raises(ValueError):
        v3.run(seeds=())
