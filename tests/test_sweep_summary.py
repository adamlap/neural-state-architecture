from pathlib import Path

import pytest

from experiments.self_state.summarize_sweep import bootstrap_ci, summarize


def test_bootstrap_ci_is_deterministic_and_bounded() -> None:
    values = [-0.2, 0.1, 0.3, 0.4]
    first = bootstrap_ci(values, seed=42, samples=2000)
    second = bootstrap_ci(values, seed=42, samples=2000)

    assert first == second
    assert first[0] <= sum(values) / len(values) <= first[1]


def test_bootstrap_ci_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([], seed=1)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], seed=1, samples=0)


def test_summary_contains_uncertainty_by_perturbation(tmp_path: Path) -> None:
    artifact = tmp_path / "sweep.json"
    artifact.write_text(
        '{\n'
        '  "seed": 1,\n'
        '  "results": [\n'
        '    {"perturbation": 1.0, "recovery_advantage": 0.2, "auc_advantage": -0.1},\n'
        '    {"perturbation": 2.0, "recovery_advantage": 0.4, "auc_advantage": 0.1}\n'
        '  ],\n'
        '  "finite": true\n'
        '}\n',
        encoding="utf-8",
    )

    result = summarize([artifact])

    assert result["seed_count"] == 1
    assert "recovery_advantage_ci95" in result
    assert "auc_advantage_ci95" in result
    assert len(result["by_perturbation"]) == 2
    assert "recovery_advantage_ci95" in result["by_perturbation"][0]
    assert "auc_advantage_ci95" in result["by_perturbation"][0]
