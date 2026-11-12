import pytest

from experiments.organic import dog_test_v4 as v4


def test_no_single_trace_covers_delays_from_3_to_150():
    fast = v4.run_one("single_0.8", 0, n_events=400)
    assert fast["by_delay"]["3"] > 0.8 and fast["by_delay"]["150"] < 0.5


def test_learners_run_and_report_every_delay():
    r = v4.run_one("snr_gated", 1, n_events=200)
    assert set(r["by_delay"]) <= {str(d) for d in v4.DELAYS} and 0.0 <= r["valenced"] <= 1.0 and r["decoy"] >= 0.0


def test_run_validates_seeds():
    with pytest.raises(ValueError):
        v4.run(seeds=())
