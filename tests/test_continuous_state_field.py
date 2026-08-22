from __future__ import annotations

import time

import torch

from nsa.runtime.continuous_state_field import ContinuousStateField


def test_step_uses_measured_elapsed_time() -> None:
    field = ContinuousStateField(torch.tensor([1.0]), lambda state, _: torch.ones_like(state), enabled=True)

    assert field.step_now(10.0) is False  # establishes the wall-clock origin
    assert field.step_now(10.25) is True
    assert torch.allclose(field.state, torch.tensor([1.25]))
    assert field.status().last_dt == 0.25


def test_async_input_is_delivered_to_field() -> None:
    seen = []

    def dynamics(state, external):
        seen.append(external)
        return torch.zeros_like(state)

    field = ContinuousStateField(torch.zeros(1), dynamics, enabled=True)
    field.inject("speech:event")

    assert field.step_now(1.0) is False
    assert field.status().pending_inputs == 1
    assert field.step_now(1.1) is True

    assert seen == ["speech:event"]
    assert field.status().pending_inputs == 0


def test_async_input_reducer_can_preserve_multiple_events() -> None:
    seen = []

    def dynamics(state, external):
        seen.append(external)
        return torch.zeros_like(state)

    field = ContinuousStateField(
        torch.zeros(1),
        dynamics,
        enabled=True,
        input_reducer=lambda events: "|".join(events) if events else None,
    )
    field.inject("speech:hello")
    field.inject("vision:person")

    assert field.step_now(1.0) is False
    assert field.step_now(1.1) is True
    assert seen == ["speech:hello|vision:person"]


def test_background_field_evolves_without_external_input() -> None:
    field = ContinuousStateField(
        torch.zeros(1),
        lambda state, _: torch.ones_like(state),
        integration_cadence_seconds=0.005,
        enabled=True,
    )

    assert field.start() is True
    time.sleep(0.035)
    assert field.stop(timeout=1.0) is True

    assert field.status().integration_count > 0
    assert field.status().elapsed_seconds > 0.0
    assert float(field.state.item()) > 0.0


def test_nonfinite_derivative_fails_closed() -> None:
    field = ContinuousStateField(
        torch.zeros(1),
        lambda state, _: torch.full_like(state, float("nan")),
        enabled=True,
    )

    field.step_now(1.0)
    assert field.step_now(1.1) is False
    status = field.status()
    assert status.enabled is False
    assert status.last_error is not None
    assert torch.equal(field.state, torch.zeros(1))
