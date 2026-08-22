from experiments.live.cce_predictive_multiseed import TASKS, run


def test_multiseed_validation_is_finite_and_beats_persistence() -> None:
    result = run(7, epochs=40)
    assert len(result["tasks"]) == len(TASKS)
    assert result["all_tasks_beat_persistence"] is True
    for row in result["tasks"]:
        assert row["mse"] >= 0.0
        assert row["persistence_mse"] >= 0.0
        assert row["beats_persistence"] is True
