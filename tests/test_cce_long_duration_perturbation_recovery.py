import torch

from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_persistent_state_recovers_after_perturbation():
    state = PersistentCognitiveState(4, decay=0.8, learning_rate=1.0)
    zero = torch.zeros(4)
    impulse = torch.ones(4) * 2.0

    baseline = state.observe(zero, dt=0.05)
    perturbed = state.observe(impulse, dt=0.05)
    assert torch.linalg.vector_norm(perturbed.working) > torch.linalg.vector_norm(baseline.working)

    for _ in range(20):
        state.observe(zero, dt=0.05)

    final = state.snapshot()
    assert torch.linalg.vector_norm(final.working) < torch.linalg.vector_norm(perturbed.working)
    assert 0.0 <= final.uncertainty <= 1.0
    assert torch.isfinite(final.working).all()
    assert torch.isfinite(final.self_state).all()
    assert torch.isfinite(final.goal).all()
