from __future__ import annotations

from experiments.live.cce_continuous_dynamics import run


def test_input_free_dynamics_and_async_perturbation() -> None:
    result = run(duration=0.25, cadence=0.01, perturb_at=0.1)

    assert result["no_input_evolved"] is True
    assert result["perturbation_injected"] is True
    assert result["perturbation_observed"] is True
    assert result["running_after_stop"] is False
    assert result["last_error"] is None
    assert result["integration_count"] > 0
