import torch

from experiments.self_state.tnc_bridge import TNCStateFeedback


def test_bridge_uses_legal_projected_transition():
    bridge = TNCStateFeedback(state_dim=7)
    projected = bridge.projected_transition()
    assert torch.allclose(projected, torch.tril(projected))
    assert torch.all(projected.diagonal() >= 0)


def test_bridge_preserves_batch_shape():
    bridge = TNCStateFeedback(state_dim=7)
    state = torch.randn(4, 7)
    assert bridge(state).shape == state.shape
