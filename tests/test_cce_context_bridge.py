from __future__ import annotations

import torch

from nsa.runtime.cce_context_bridge import CognitiveContextBridge, CognitiveContextEnvelope
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def test_context_envelope_is_immutable_and_structured() -> None:
    state = PersistentCognitiveState(3)
    snapshot = state.observe(torch.tensor([1.0, 2.0, 3.0]), dt=1.0)
    envelope = CognitiveContextBridge.envelope(snapshot)
    assert isinstance(envelope, CognitiveContextEnvelope)
    assert envelope.to_dict()["update_count"] == 1
    assert envelope.to_json().startswith("{")


def test_render_prompt_declares_read_only_boundary() -> None:
    state = PersistentCognitiveState(2)
    snapshot = state.observe(torch.tensor([0.5, 0.25]), dt=1.0)
    prompt = CognitiveContextBridge.render_prompt(snapshot, "observe")
    assert "READ-ONLY" in prompt
    assert "CCE_SOFT_STATE_JSON=" in prompt
    assert "TASK=observe" in prompt


def test_nonfinite_state_is_rejected() -> None:
    state = PersistentCognitiveState(2)
    snapshot = state.observe(torch.tensor([0.5, 0.25]), dt=1.0)
    bad = type(snapshot)(
        working=torch.tensor([float("nan"), 0.0]),
        self_state=snapshot.self_state,
        goal=snapshot.goal,
        uncertainty=snapshot.uncertainty,
        elapsed_seconds=snapshot.elapsed_seconds,
        update_count=snapshot.update_count,
    )
    try:
        CognitiveContextBridge.envelope(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite state must be rejected")
