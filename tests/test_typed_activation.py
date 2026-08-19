import json

import pytest
import torch

from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.core.typed_activation import (
    CanonicalTypedActivation,
    HardStateMutationError,
)
from nsa.epistemic import EpistemicTier, EpistemicVector


def make_activation() -> CanonicalTypedActivation:
    state = UnifiedCognitiveState(
        semantic_state=torch.zeros(1, 4),
        operational_self_state=torch.zeros(1, 4),
        epistemic_state=EpistemicVector(0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, EpistemicTier.UNVERIFIED),
        authority_state=torch.tensor([0.25]),
        provenance_state=ProvenanceRecord("p0", "backend://test", "abc", 1.0),
        temporal_state=TemporalHorizonState(0, 10, 0.0),
        goal_state=TeleologicalState("test", 0.5, 0.0),
    )
    return CanonicalTypedActivation(state)


def test_model_cannot_write_hard_authority_state() -> None:
    activation = make_activation()
    assert activation.can_write("authority_state", "model") is False
    with pytest.raises(HardStateMutationError):
        activation.model_proposal("authority_state", torch.tensor([1.0]))


def test_runtime_commit_returns_new_activation_without_mutating_original() -> None:
    activation = make_activation()
    updated = activation.runtime_commit("operational_self_state", torch.ones(1, 4))

    assert torch.equal(activation.state.operational_self_state, torch.zeros(1, 4))
    assert torch.equal(updated.state.operational_self_state, torch.ones(1, 4))
    assert activation.state.authority_state.item() == updated.state.authority_state.item()


def test_serialization_is_json_compatible_and_versioned() -> None:
    payload = make_activation().to_dict()
    encoded = json.dumps(payload)

    assert "schema_version" in payload
    assert payload["schema_version"] == "1.0"
    assert "authority_state" in payload["state"]
    assert "UNVERIFIED" in encoded
