import pytest

pytest.importorskip("numpy")

from experiments.organic import runtime_stream_test as rs


def test_runtime_learning_matches_the_standalone_dog_test_at_short_delay():
    r = rs.learning(0, 0.8, 3, n_events=150, checkpoints=(75, 150))
    assert r["final"]["valenced_accuracy"] > 0.8 and r["final"]["decoy_accuracy"] > 0.9


def test_spoofed_sensor_is_ignored_when_gated_and_harmful_when_not():
    g = rs.spoof_run(0, True, n_events=200)
    u = rs.spoof_run(0, False, n_events=200)
    assert g["valenced_accuracy"] > 0.8 and g["learning_blocked"] > 0
    assert u["valenced_accuracy"] < g["valenced_accuracy"] - 0.3


def test_rank_corr_and_validation():
    assert rs.rank_corr([1, 2, 3], [3, 6, 9]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rs.run(seeds=())
