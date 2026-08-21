from experiments.self_model.aggregate_evaluations import aggregate


def _row(seed: int, predictor: float, persistence: float) -> dict:
    return {
        "seed": seed,
        "test_predictor_mse": predictor,
        "test_persistence_mse": persistence,
        "test_mse_improvement": persistence - predictor,
        "predictor_beats_persistence": predictor < persistence,
        "finite": True,
    }


def test_aggregate_reports_cross_seed_effect() -> None:
    result = aggregate([
        _row(1, 0.25, 0.50),
        _row(2, 0.40, 0.50),
        _row(3, 0.60, 0.50),
    ])

    assert result["evaluations"] == 3
    assert result["seeds"] == [1, 2, 3]
    assert result["mean_mse_improvement"] == (0.25 + 0.10 - 0.10) / 3
    assert result["positive_improvement_fraction"] == 2 / 3
    assert result["predictor_win_fraction"] == 2 / 3
    assert result["all_finite"] is True


def test_aggregate_does_not_turn_non_wins_into_wins() -> None:
    result = aggregate([_row(7, 0.8, 0.5)])
    assert result["mean_mse_improvement"] < 0
    assert result["predictor_win_fraction"] == 0.0
