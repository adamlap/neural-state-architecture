import pytest
import torch

from nsa.runtime.cce_governed_feedback import (
    CognitiveFeedbackProposal,
    GovernedCognitiveFeedback,
)
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_feedback_is_bounded_and_persistent():
    state = PersistentCognitiveState(2, learning_rate=1.0)
    gate = GovernedCognitiveFeedback(state, max_norm=0.1)
    result = gate.apply(
        CognitiveFeedbackProposal((10.0, 0.0), (0.0, 10.0), confidence=1.0, source="test"),
        dt=1.0,
    )
    assert result.accepted
    assert result.clipped_norm <= 0.100001
    assert result.snapshot.update_count == 1
    assert torch.isfinite(result.snapshot.working).all()


def test_confidence_scales_untrusted_feedback():
    state = PersistentCognitiveState(2, learning_rate=1.0)
    gate = GovernedCognitiveFeedback(state, max_norm=1.0)
    result = gate.apply(
        CognitiveFeedbackProposal((1.0, 0.0), confidence=0.25, source="test"),
        dt=1.0,
    )
    assert torch.allclose(result.snapshot.working, torch.tensor([0.25, 0.0]))


def test_nonfinite_and_invalid_feedback_fails_closed():
    state = PersistentCognitiveState(2)
    gate = GovernedCognitiveFeedback(state)
    with pytest.raises(ValueError):
        gate.apply(CognitiveFeedbackProposal((float("nan"), 0.0), confidence=1.0), dt=1.0)
    with pytest.raises(ValueError):
        gate.apply(CognitiveFeedbackProposal((0.0, 0.0), confidence=2.0), dt=1.0)
