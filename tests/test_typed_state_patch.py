import pytest
import torch

from nsa.core.typed_state_patch import CanonicalStatePatch, StatePatchConflict
from nsa.core.typed_activation import HardStateMutationError
from tests.test_typed_activation import make_activation


def test_sparse_patch_composes_disjoint_updates() -> None:
    first = CanonicalStatePatch({"semantic_state": torch.ones(1, 4)})
    second = CanonicalStatePatch({"operational_self_state": torch.full((1, 4), 2.0)})

    merged = first.compose(second)
    assert set(merged.values) == {"semantic_state", "operational_self_state"}


def test_conflicting_patch_composition_is_rejected() -> None:
    first = CanonicalStatePatch({"semantic_state": torch.zeros(1, 2)})
    second = CanonicalStatePatch({"semantic_state": torch.ones(1, 2)})

    with pytest.raises(StatePatchConflict):
        first.compose(second)


def test_runtime_application_preserves_hard_state() -> None:
    activation = make_activation()
    patch = CanonicalStatePatch(
        {
            "semantic_state": torch.ones(1, 4),
            "operational_self_state": torch.full((1, 4), 2.0),
        }
    )

    updated = patch.apply_runtime(activation)
    assert torch.equal(updated.state.semantic_state, torch.ones(1, 4))
    assert torch.equal(updated.state.operational_self_state, torch.full((1, 4), 2.0))
    assert torch.equal(updated.state.authority_state, activation.state.authority_state)


def test_model_proposals_cannot_cross_hard_state_boundary() -> None:
    activation = make_activation()
    patch = CanonicalStatePatch({"authority_state": torch.tensor([0.99])})

    with pytest.raises(HardStateMutationError):
        patch.model_proposals(activation)


def test_unknown_partial_state_field_is_rejected() -> None:
    with pytest.raises(KeyError):
        CanonicalStatePatch({"not_a_state_field": 1})
