from enum import IntEnum

from nsa.core.heterogeneous_algebra import BooleanDomain, EnumDomain, HeterogeneousState, NumericRangeDomain
from nsa.core.transition_cone import TransitionCone, TransitionDirection


class Level(IntEnum):
    LOW = 0
    HIGH = 1


def make_state(flag: bool, risk: float, level: Level) -> HeterogeneousState:
    return HeterogeneousState(
        (flag, risk, level),
        (BooleanDomain(), NumericRangeDomain(), EnumDomain(Level)),
    )


def test_increasing_decreasing_and_immutable_coordinates():
    source = make_state(False, 0.8, Level.LOW)
    safe = make_state(True, 0.2, Level.HIGH)
    cone = TransitionCone(
        (TransitionDirection.INCREASE, TransitionDirection.DECREASE, TransitionDirection.INCREASE)
    )
    assert cone.allows(source, safe)
    assert cone.is_projected(source, safe)


def test_illegal_target_is_rejected():
    source = make_state(False, 0.2, Level.LOW)
    illegal = make_state(True, 0.9, Level.HIGH)
    cone = TransitionCone(
        (TransitionDirection.INCREASE, TransitionDirection.DECREASE, TransitionDirection.INCREASE)
    )
    assert not cone.allows(source, illegal)


def test_exact_projection_recovers_legal_boundary():
    source = make_state(False, 0.4, Level.HIGH)
    candidate = make_state(True, 0.9, Level.LOW)
    cone = TransitionCone(
        (TransitionDirection.INCREASE, TransitionDirection.DECREASE, TransitionDirection.UNCHANGED)
    )
    projected = cone.project(source, candidate)
    assert projected.values == (True, 0.4, Level.HIGH)
    assert cone.allows(source, projected)
    assert cone.project(source, projected) == projected


def test_projection_is_idempotent():
    source = make_state(True, 0.5, Level.LOW)
    candidate = make_state(False, 0.1, Level.HIGH)
    cone = TransitionCone(
        (TransitionDirection.INCREASE, TransitionDirection.DECREASE, TransitionDirection.INCREASE)
    )
    projected = cone.project(source, candidate)
    assert cone.project(source, projected) == projected


def test_cone_arity_is_checked():
    source = make_state(False, 0.5, Level.LOW)
    target = make_state(True, 0.4, Level.HIGH)
    cone = TransitionCone((TransitionDirection.INCREASE,))
    try:
        cone.allows(source, target)
    except ValueError as exc:
        assert "arity" in str(exc)
    else:
        raise AssertionError("expected transition cone arity validation")
