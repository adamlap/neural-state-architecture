from experiments.self_model.bootstrap_analysis import analyze, bootstrap_mean_ci


def test_bootstrap_is_deterministic() -> None:
    values = [0.2, 0.1, 0.3, -0.05]
    assert bootstrap_mean_ci(values, samples=1000, seed=7) == bootstrap_mean_ci(values, samples=1000, seed=7)


def test_analysis_reports_fraction_and_ci() -> None:
    result = analyze([0.2, 0.1, 0.3], samples=1000, seed=1)
    assert result["evaluations"] == 3
    assert result["fraction_positive"] == 1.0
    assert result["ci_excludes_zero"] is True


def test_analysis_rejects_nonfinite() -> None:
    try:
        analyze([0.1, float("nan")])
    except ValueError:
        return
    raise AssertionError("non-finite improvements must be rejected")
