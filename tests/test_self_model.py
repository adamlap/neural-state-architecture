import torch

from nsa.self_model import CapabilityMonitor, PredictiveSelfState, SelfRegulationController


def test_predictive_self_state_shapes_and_error():
    module = PredictiveSelfState(d_model=16, state_dim=7)
    meaning = torch.randn(2, 5, 16)
    state = torch.randn(2, 5, 7)
    actual = torch.randn(2, 5, 7)
    out = module(meaning, state, actual)
    assert out["predicted_state"].shape == actual.shape
    assert out["prediction_error"].shape == actual.shape
    assert out["error_signal"].shape == actual.shape
    assert out["prediction_mse"].shape == (2, 5, 1)
    assert torch.isfinite(out["prediction_error"]).all()


def test_self_regulation_requests_reassessment_on_large_error():
    controller = SelfRegulationController()
    small = controller(torch.zeros(2, 3, 7))
    large = controller(torch.full((2, 3, 7), 10.0))
    assert not small.request_reassessment.any()
    assert large.request_reassessment.all()
    assert torch.all(large.confidence < small.confidence)


def test_capability_monitor_is_bounded():
    monitor = CapabilityMonitor(d_model=16, state_dim=7)
    score = monitor(torch.randn(4, 8, 16), torch.randn(4, 8, 7))
    assert score.shape == (4, 8, 1)
    assert torch.all((score >= 0) & (score <= 1))
