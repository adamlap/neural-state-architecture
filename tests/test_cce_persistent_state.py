from __future__ import annotations

import torch

from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_state_persists_and_changes_with_observations() -> None:
    state = PersistentCognitiveState(3, learning_rate=1.0, decay=0.5)
    first = state.observe(torch.tensor([1.0, 0.0, 0.0]), dt=1.0)
    second = state.observe(torch.tensor([0.0, 1.0, 0.0]), dt=1.0)

    assert first.update_count == 1
    assert second.update_count == 2
    assert not torch.allclose(first.working, second.working)
    assert torch.linalg.vector_norm(second.self_state).item() > 0.0
    assert second.elapsed_seconds == 2.0


def test_target_channel_is_persistent_and_optional() -> None:
    state = PersistentCognitiveState(2, learning_rate=1.0)
    state.observe(torch.tensor([0.2, 0.4]), dt=1.0, target=torch.tensor([1.0, 0.0]))
    snapshot = state.observe(torch.tensor([0.2, 0.4]), dt=1.0)

    assert torch.allclose(snapshot.goal, torch.tensor([1.0, 0.0]))
    assert 0.0 <= snapshot.uncertainty <= 1.0


def test_nonfinite_observation_fails_closed() -> None:
    state = PersistentCognitiveState(2)
    try:
        state.observe(torch.tensor([float("nan"), 0.0]), dt=1.0)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite observation was accepted")
