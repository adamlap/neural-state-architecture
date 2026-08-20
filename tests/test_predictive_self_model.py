from __future__ import annotations

import pytest
import torch

from nsa.predictive_self_model import PredictiveSelfModel
from nsa.self_state.model import SelfState


def test_prediction_is_bounded_and_advances_logical_step() -> None:
    model = PredictiveSelfModel(action_dim=2, hidden_dim=16)
    state = SelfState(confidence=0.7, uncertainty=0.2, step=4)
    result = model.predict(state, action=[0.25, 0.8])

    assert result.predicted.step == 5
    for field in (
        "confidence",
        "uncertainty",
        "perceived_risk",
        "capability_awareness",
        "resource_pressure",
        "goal_progress",
        "state_prediction_error",
    ):
        value = getattr(result.predicted, field)
        assert 0.0 <= value <= 1.0


def test_training_loss_is_finite_and_differentiable() -> None:
    model = PredictiveSelfModel(action_dim=1)
    state = torch.full((4, model.state_dim), 0.5)
    target = torch.full((4, model.state_dim), 0.75)
    action = torch.zeros((4, 1))

    loss = model.training_loss(state, target, action)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_prediction_error_compares_explicit_state_only() -> None:
    model = PredictiveSelfModel()
    state = SelfState(confidence=0.5, uncertainty=0.5)
    observed = SelfState(confidence=0.9, uncertainty=0.1)

    result = model.predict(state).compare(observed)
    assert result.mse is not None
    assert result.mse >= 0.0


def test_action_dimension_and_input_validation() -> None:
    model = PredictiveSelfModel(action_dim=2)
    state = torch.zeros((1, model.state_dim))

    with pytest.raises(ValueError, match="action is required"):
        model(state)
    with pytest.raises(ValueError, match="action length"):
        model.predict(SelfState(), action=[0.1])
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[float("nan")] * model.state_dim]), torch.zeros((1, 2)))
