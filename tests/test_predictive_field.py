import torch

from nsa.runtime.predictive_dynamics import StatePredictor
from nsa.runtime.predictive_field import PredictiveDynamicsField


def test_predictive_field_is_disabled_by_default():
    predictor = StatePredictor(2)
    field = PredictiveDynamicsField(predictor)
    state = torch.tensor([[1.0, 2.0]])
    assert torch.equal(field(state), torch.zeros_like(state))


def test_predictive_field_converts_next_state_to_derivative():
    predictor = StatePredictor(2)
    field = PredictiveDynamicsField(predictor, reference_dt=0.5, enabled=True)
    state = torch.tensor([[1.0, 2.0]])
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
        predictor.net[-1].bias.copy_(torch.tensor([2.0, 4.0]))
    derivative = field(state)
    expected = (torch.tensor([[2.0, 4.0]]) - state) / 0.5
    assert torch.allclose(derivative, expected)


def test_external_input_reaches_predictor():
    predictor = StatePredictor(2, input_dim=1)
    field = PredictiveDynamicsField(predictor, enabled=True)
    state = torch.zeros(1, 2)
    external = torch.ones(1, 1)
    derivative = field(state, external)
    assert derivative.shape == state.shape
    assert torch.isfinite(derivative).all()
