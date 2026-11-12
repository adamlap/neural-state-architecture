import pytest

from experiments.organic import dual_process_test as dp


def test_policy_runs_report_online_correctness_and_query_counts():
    always = dp.run_policy("always", 0)
    never = dp.run_policy("never", 0)
    assert always["queries"] == dp.N_EVENTS and never["queries"] == 0 and len(always["ok"]) == dp.N_EVENTS
    a, n = dp.window(always["ok"], 200, dp.SHIFT, always["valenced"]), dp.window(never["ok"], 200, dp.SHIFT, never["valenced"])
    assert a > 0.9 and n < 0.2                                  # on valenced cues a learner is right, a never-queried one is not (it predicts "nothing")
    assert dp.window(never["ok"], 200, dp.SHIFT) > 0.6         # ...while overall accuracy hides this, because decoys are 2/3 of events


def test_gated_policy_uses_far_fewer_queries_than_always_and_still_learns():
    g = dp.run_policy("gated", 1)
    assert g["queries"] < 0.6 * dp.N_EVENTS and dp.window(g["ok"], 200, dp.SHIFT, g["valenced"]) > 0.7


def test_recovery_helper_and_validation():
    assert dp.recovery([True] * dp.N_EVENTS) == 0 and dp.recovery([False] * dp.N_EVENTS) is None
    with pytest.raises(ValueError):
        dp.run(seeds=())
