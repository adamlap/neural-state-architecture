import torch

from nsa.self_state_loop import SelfStateRegulator


def test_regulator_cannot_modify_hard_security_coordinate():
    torch.manual_seed(0)
    regulator = SelfStateRegulator(state_dim=8)
    state = torch.randn(2, 5, 8)
    error = torch.randn_like(state)
    regulated = regulator(state, error, enabled=True)
    assert torch.equal(regulated[..., 0], state[..., 0])


def test_regulator_is_ablatable():
    torch.manual_seed(0)
    regulator = SelfStateRegulator(state_dim=8)
    state = torch.randn(2, 5, 8)
    error = torch.randn_like(state)
    assert torch.equal(regulator(state, error, enabled=False), state)


def test_regulator_delta_is_bounded():
    torch.manual_seed(0)
    regulator = SelfStateRegulator(state_dim=8, max_delta=0.25)
    state = torch.randn(2, 5, 8)
    error = torch.randn_like(state) * 100
    delta = regulator(state, error) - state
    assert float(delta[..., 1:].abs().max()) <= 0.25 + 1e-6
