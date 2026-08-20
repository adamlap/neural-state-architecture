from experiments.transitions.capability_cost import run


def test_transition_cost_benchmark_is_finite_and_normalized() -> None:
    result = run(seed=7, samples=100)
    assert result["finite"] is True
    assert result["samples"] == 100
    assert 0.0 <= result["projected_fraction"] <= 1.0
    assert 0.0 <= result["exact_legal_fraction"] <= 1.0
    assert result["constrained_ns_per_transition"] >= 0.0
    assert result["unconstrained_ns_per_transition"] >= 0.0


def test_transition_engine_can_project_illegal_candidates() -> None:
    result = run(seed=42, samples=200)
    assert result["projected_fraction"] > 0.0
    assert result["max_numeric_distortion"] >= result["mean_numeric_distortion"]
