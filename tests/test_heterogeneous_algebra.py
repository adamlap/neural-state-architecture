from __future__ import annotations

from enum import IntEnum

import pytest

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    CapabilityDomain,
    ConstraintSetDomain,
    EnumDomain,
    HeterogeneousState,
    NumericRangeDomain,
    ProbabilityInterval,
    ProbabilityIntervalDomain,
    TemporalWindow,
    TemporalWindowDomain,
)


class Level(IntEnum):
    LOW = 0
    HIGH = 1


def make_state(flag: bool, risk: float, caps: frozenset[str], level: Level):
    return HeterogeneousState(
        (flag, risk, caps, level),
        (BooleanDomain(), NumericRangeDomain(), CapabilityDomain(), EnumDomain(Level)),
    )


def test_product_join_and_meet_preserve_domain_specific_semantics():
    left = make_state(False, 0.2, frozenset({"read"}), Level.LOW)
    right = make_state(True, 0.8, frozenset({"write"}), Level.HIGH)
    assert left.join(right).values == (True, 0.8, frozenset({"read", "write"}), Level.HIGH)
    assert left.meet(right).values == (False, 0.2, frozenset(), Level.LOW)


def test_product_order_is_coordinate_wise():
    low = make_state(False, 0.2, frozenset(), Level.LOW)
    high = make_state(True, 0.8, frozenset({"read"}), Level.HIGH)
    assert low.leq(high)
    assert not high.leq(low)


def test_join_and_meet_are_idempotent():
    state = make_state(True, 0.4, frozenset({"read"}), Level.HIGH)
    assert state.join(state) == state
    assert state.meet(state) == state


def test_incompatible_products_are_rejected():
    left = make_state(False, 0.2, frozenset(), Level.LOW)
    right = HeterogeneousState((False, 0.2), (BooleanDomain(), NumericRangeDomain()))
    with pytest.raises(ValueError):
        left.join(right)


def test_equivalent_domain_instances_are_compatible():
    a = HeterogeneousState((0.2,), (NumericRangeDomain(),))
    b = HeterogeneousState((0.8,), (NumericRangeDomain(),))
    assert a.join(b).values == (0.8,)


def test_numeric_domain_rejects_nan_and_out_of_range_values():
    domain = NumericRangeDomain()
    with pytest.raises(ValueError):
        domain.validate(float("nan"))
    with pytest.raises(ValueError):
        domain.validate(1.1)


def test_constraint_domain_is_a_compositional_powerset():
    domain = ConstraintSetDomain()
    left = frozenset({"no_external_write", "human_approval"})
    right = frozenset({"human_approval", "budget_limit"})
    assert domain.join(left, right) == frozenset({"no_external_write", "human_approval", "budget_limit"})
    assert domain.meet(left, right) == frozenset({"human_approval"})


def test_probability_interval_lattice_has_explicit_bottom():
    domain = ProbabilityIntervalDomain()
    a = ProbabilityInterval(0.2, 0.6)
    b = ProbabilityInterval(0.5, 0.9)
    assert domain.join(a, b) == ProbabilityInterval(0.2, 0.9)
    assert domain.meet(a, b) == ProbabilityInterval(0.5, 0.6)
    bottom = domain.meet(ProbabilityInterval(0.0, 0.1), ProbabilityInterval(0.9, 1.0))
    assert bottom.is_empty
    assert domain.join(bottom, a) == a


def test_probability_interval_rejects_invalid_bounds():
    domain = ProbabilityIntervalDomain()
    with pytest.raises(ValueError):
        domain.validate(ProbabilityInterval(-0.1, 0.5))
    with pytest.raises(ValueError):
        domain.validate(ProbabilityInterval(0.5, 1.1))
    with pytest.raises(ValueError):
        domain.validate(ProbabilityInterval(0.5, 0.4))


def test_temporal_window_lattice_has_explicit_bottom():
    domain = TemporalWindowDomain()
    a = TemporalWindow(10.0, 20.0)
    b = TemporalWindow(15.0, 30.0)
    assert domain.join(a, b) == TemporalWindow(10.0, 30.0)
    assert domain.meet(a, b) == TemporalWindow(15.0, 20.0)
    bottom = domain.meet(TemporalWindow(0.0, 1.0), TemporalWindow(2.0, 3.0))
    assert bottom.is_empty
    assert domain.join(bottom, a) == a


def test_heterogeneous_product_can_mix_new_domains():
    state = HeterogeneousState(
        (ProbabilityInterval(0.2, 0.7), TemporalWindow(100.0, 200.0), frozenset({"human_approval"})),
        (ProbabilityIntervalDomain(), TemporalWindowDomain(), ConstraintSetDomain()),
    )
    other = HeterogeneousState(
        (ProbabilityInterval(0.5, 0.9), TemporalWindow(150.0, 250.0), frozenset({"budget_limit"})),
        (ProbabilityIntervalDomain(), TemporalWindowDomain(), ConstraintSetDomain()),
    )
    assert state.join(other).values == (
        ProbabilityInterval(0.2, 0.9),
        TemporalWindow(100.0, 250.0),
        frozenset({"human_approval", "budget_limit"}),
    )
