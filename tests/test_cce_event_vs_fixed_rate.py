from __future__ import annotations

import torch

from experiments.live.cce_event_vs_fixed_rate import _field


def test_control_field_is_dynamic_and_input_conditioned() -> None:
    state = torch.tensor([1.0, -0.5])
    without_input = _field(state, None)
    with_input = _field(state, 0.75)

    assert torch.allclose(without_input, -0.25 * state)
    assert not torch.allclose(without_input, with_input)
    assert torch.isfinite(with_input).all()
