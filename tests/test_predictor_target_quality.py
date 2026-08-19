from experiments.self_state.predictor_target_quality import run


def test_predictor_target_quality_is_finite_and_secure() -> None:
    result = run(42)
    assert result["finite"] is True
    assert result["summary"]["max_security_delta"] == 0.0
    assert len(result["results"]) == len(result["perturbations"])


def test_predictor_target_quality_reports_directional_alignment() -> None:
    result = run(42)
    assert all("correction_oracle_cosine" in row for row in result["results"])
    assert all(-1.0 <= row["correction_oracle_cosine"] <= 1.0 for row in result["results"])
