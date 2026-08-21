import pytest
from .dynamics import DynamicalState


def test_state_evolves_without_external_input():
    state = DynamicalState(0.8, 0.2, 0.1)
    before = state.vector()
    state.tick()
    assert state.vector() != before


def test_state_is_bounded():
    state = DynamicalState()
    for _ in range(1000):
        state.tick(drive=100.0)
    assert all(-1.0 <= x <= 1.0 for x in state.vector())


def test_invalid_dt_is_rejected():
    with pytest.raises(ValueError):
        DynamicalState().tick(dt=0)
