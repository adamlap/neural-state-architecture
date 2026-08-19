from experiments.self_state.regulator_gain_sweep import run


def test_regulator_gain_sweep_is_finite_and_preserves_security() -> None:
    result = run(
        seed=1,
        gains=[0.1],
        max_deltas=[0.1],
        perturbations=[1.0],
        steps=2,
    )
    assert result["finite"] is True
    assert result["summary"]["all_security_immutable"] is True
    assert len(result["results"]) == 1
    assert "auc_advantage" in result["results"][0]
