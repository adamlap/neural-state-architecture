from experiments.self_state.local_contraction import run


def test_local_contraction_preserves_hard_security_coordinate() -> None:
    result = run(42)
    assert result["finite"] is True
    assert result["summary"]["max_security_delta"] == 0.0
    assert len(result["results"]) == 7


def test_local_contraction_moves_toward_self_model_prediction() -> None:
    """The regulator is defined to contract prediction error, not native drift.

    The native trajectory is an independent reference measurement; the
    regulator has no information about it and therefore must not be tested as
    though it were its target.  Its explicit architectural target is the
    self-model prediction.
    """
    result = run(42)
    row = result["results"][0]
    assert row["prediction_distance_after"] < row["prediction_distance_before"]


def test_local_contraction_respects_max_delta() -> None:
    result = run(42, max_delta=0.1)
    assert max(row["max_soft_delta"] for row in result["results"]) <= 0.1 + 1e-6
