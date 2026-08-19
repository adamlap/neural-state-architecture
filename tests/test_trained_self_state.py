import torch
import pytest

from experiments.self_state.trained_regulation import train_predictor
from nsa.cognitive import NSACognitiveLM


def _model() -> NSACognitiveLM:
    return NSACognitiveLM(
        vocab_size=32,
        d_model=16,
        state_dim=4,
        num_layers=1,
        num_heads=4,
        max_seq_len=8,
        dropout=0.0,
    )


def test_train_predictor_reduces_native_state_prediction_error() -> None:
    torch.manual_seed(7)
    model = _model()
    model.eval()
    tokens = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        _, hidden, states = model.nsa(tokens)
        inputs = hidden[:, :-1]
        previous = states[:, :-1]
        target = states[:, 1:]
        before = (model.self_model.predict(inputs, previous) - target).pow(2).mean()

    train_predictor(model, hidden, states, epochs=25, learning_rate=1e-2)
    with torch.no_grad():
        after = (model.self_model.predict(inputs, previous) - target).pow(2).mean()

    assert float(after) < float(before)
    assert all(not parameter.requires_grad for parameter in model.nsa.parameters())


def test_train_predictor_rejects_invalid_configuration() -> None:
    model = _model()
    hidden = torch.zeros(2, 8, 16)
    states = torch.zeros(2, 8, 4)
    with pytest.raises(ValueError):
        train_predictor(model, hidden, states, epochs=0)
    with pytest.raises(ValueError):
        train_predictor(model, hidden, states, learning_rate=0.0)
