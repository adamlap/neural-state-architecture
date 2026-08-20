from frozenset import frozenset

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    HeterogeneousState,
    NumericRangeDomain,
)
from nsa.core.transition_cone import TransitionCone, TransitionDirection
from nsa.transitions.engine import TransitionEngine


def _state(value: float, enabled: bool = True) -> HeterogeneousState:
    return HeterogeneousState(
        values=(enabled, value),
        domains=(BooleanDomain(), NumericRangeDomain(0.0, 1.0)),
    )


def test_legal_heterogeneous_transition_is_accepted_without_projection() -> None:
    source = _state(0.4, enabled=False)
    target = _state(0.7, enabled=False)
    cone = TransitionCone(
        (TransitionDirection.UNCHANGED, TransitionDirection.INCREASE)
    )

    result = TransitionEngine().apply_heterogeneous(source, target, cone)

    assert result.accepted
    assert not result.projected
    assert result.state == target


def test_illegal_heterogeneous_transition_is_projected_coordinate_wise() -> None:
    source = _state(0.4, enabled=True)
    candidate = _state(0.1, enabled=False)
    cone = TransitionCone(
        (TransitionDirection.UNCHANGED, TransitionDirection.INCREASE)
    )

    result = TransitionEngine().apply_heterogeneous(source, candidate, cone)

    assert result.accepted
    assert result.projected
    assert result.state.values == (True, 0.4)
    assert cone.allows(source, result.state)


def test_illegal_heterogeneous_transition_can_be_rejected_without_projection() -> None:
    source = _state(0.4, enabled=True)
    candidate = _state(0.1, enabled=False)
    cone = TransitionCone(
        (TransitionDirection.UNCHANGED, TransitionDirection.INCREASE)
    )

    result = TransitionEngine().apply_heterogeneous(
        source, candidate, cone, project_illegal=False
    )

    assert not result.accepted
    assert not result.projected
    assert result.state == source
    assert result.reason == "candidate violates transition cone"


def test_domain_mismatch_is_rejected() -> None:
    source = _state(0.4)
    candidate = HeterogeneousState(
        values=(True, 0.4),
        domains=(BooleanDomain(), NumericRangeDomain(-1.0, 1.0)),
    )
    cone = TransitionCone(
        (TransitionDirection.UNCHANGED, TransitionDirection.INCREASE)
    )

    result = TransitionEngine().apply_heterogeneous(source, candidate, cone)

    assert not result.accepted
    assert result.state == source
    assert result.reason is not None
