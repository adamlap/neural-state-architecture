import torch

from nsa.runtime.predictive_dynamics import StatePredictor, prediction_metrics, train_predictor


def test_prediction_metrics_detects_improvement():
    current = torch.tensor([[0.0], [1.0], [2.0]])
    target = current + 1.0
    predicted = target.clone()
    metrics = prediction_metrics(predicted, target, current)
    assert metrics.mse == 0.0
    assert metrics.persistence_mse > 0.0
    assert metrics.improvement == 1.0


def test_predictor_learns_deterministic_next_state():
    torch.manual_seed(7)
    states = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    targets = 0.8 * states + 0.25
    model = StatePredictor(state_dim=1, hidden_dim=24)
    metrics = train_predictor(model, states, targets, epochs=250, learning_rate=5e-3)
    assert metrics.mse < metrics.persistence_mse
    assert metrics.improvement > 0.8


def test_predictor_supports_external_input():
    torch.manual_seed(3)
    states = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    external = torch.ones(64, 1)
    targets = 0.5 * states + 0.3 * external
    model = StatePredictor(state_dim=1, input_dim=1, hidden_dim=16)
    metrics = train_predictor(model, states, targets, external, epochs=200, learning_rate=5e-3)
    assert metrics.mse < metrics.persistence_mse
