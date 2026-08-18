"""Tests for the canonical NSA typed state model."""

import pytest

from nsa.core import (
    CanonicalState,
    GoalState,
    HardState,
    ProvenanceState,
    SemanticState,
    SoftState,
    StateTransition,
)
from nsa.algebra import ConfidentialityLabel, IntegrityLabel


def make_state() -> CanonicalState:
    return CanonicalState(
        semantic=SemanticState("latent"),
        hard=HardState(
            confidentiality=ConfidentialityLabel.CONFIDENTIAL,
            integrity=IntegrityLabel.TRUSTED,
            authorizations=frozenset({"read:db"}),
            license_tier=2,
        ),
        soft=SoftState(uncertainty=0.2, risk=0.1, confidence=0.8),
        provenance=ProvenanceState(sources=("sensor-a",), evidence_ids=("e1",)),
        goals=GoalState(goals=("answer",), active_goal="answer"),
    )


def test_canonical_state_preserves_product_components():
    state = make_state()
    assert state.semantic.value == "latent"
    assert state.hard.confidentiality is ConfidentialityLabel.CONFIDENTIAL
    assert state.hard.has_permission("read:db")
    assert state.soft.confidence == 0.8
    assert state.provenance.sources == ("sensor-a",)
    assert state.goals.active_goal == "answer"


def test_semantic_update_cannot_change_hard_state():
    state = make_state()
    updated = state.with_semantic("new-latent")
    assert updated.semantic.value == "new-latent"
    assert updated.hard == state.hard


def test_soft_observation_cannot_change_hard_state():
    state = make_state()
    updated = state.observe(uncertainty=0.9, risk=0.7, confidence=0.2)
    assert updated.soft.uncertainty == 0.9
    assert updated.soft.risk == 0.7
    assert updated.hard == state.hard


def test_unauthorized_hard_transition_is_rejected():
    state = make_state()
    target = HardState(
        confidentiality=ConfidentialityLabel.PUBLIC,
        integrity=IntegrityLabel.TRUSTED,
        authorizations=state.hard.authorizations,
        license_tier=state.hard.license_tier,
    )
    with pytest.raises(PermissionError):
        state.transition(StateTransition(source=state.hard, target=target))


def test_authorized_hard_transition_requires_explicit_capability_id():
    state = make_state()
    target = HardState(confidentiality=ConfidentialityLabel.PUBLIC)
    transition = StateTransition(source=state.hard, target=target)

    with pytest.raises(ValueError):
        transition.authorize("")

    authorized = transition.authorize("cap-123", reason="approved declassification")
    updated = state.transition(authorized)
    assert updated.hard == target
    assert updated.step == state.step + 1


def test_transition_source_must_match_current_state():
    state = make_state()
    other_source = HardState(confidentiality=ConfidentialityLabel.PRIVATE)
    target = HardState(confidentiality=ConfidentialityLabel.SYSTEM)
    transition = StateTransition(
        source=other_source,
        target=target,
        authorized=True,
        capability_id="cap-123",
    )
    with pytest.raises(ValueError):
        state.transition(transition)


def test_soft_state_bounds_are_enforced():
    with pytest.raises(ValueError):
        SoftState(confidence=1.1)
    with pytest.raises(ValueError):
        SoftState(risk=-0.1)


def test_goal_state_is_not_authority():
    state = make_state()
    updated = state.with_goal(GoalState(goals=("different",), active_goal="different"))
    assert updated.goals.active_goal == "different"
    assert updated.hard == state.hard


def test_provenance_is_append_only_by_default():
    provenance = ProvenanceState(sources=("a",))
    updated = provenance.extend(source="b", transformation="model-x", evidence_id="e2")
    assert updated.sources == ("a", "b")
    assert updated.transformations == ("model-x",)
    assert updated.evidence_ids == ("e2",)


def test_join_is_conservative_for_hard_state():
    left = HardState(
        confidentiality=ConfidentialityLabel.PUBLIC,
        integrity=IntegrityLabel.TRUSTED,
        authorizations=frozenset({"a"}),
        license_tier=1,
    )
    right = HardState(
        confidentiality=ConfidentialityLabel.PRIVATE,
        integrity=IntegrityLabel.UNTRUSTED,
        authorizations=frozenset({"b"}),
        license_tier=3,
    )
    joined = left.join(right)
    assert joined.confidentiality is ConfidentialityLabel.PRIVATE
    assert joined.integrity is IntegrityLabel.UNTRUSTED
    assert joined.authorizations == frozenset({"a", "b"})
    assert joined.license_tier == 3
