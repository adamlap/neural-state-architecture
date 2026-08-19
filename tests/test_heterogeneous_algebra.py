from dataclasses import dataclass
from enum import IntEnum

import pytest

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    CapabilityDomain,
    EnumDomain,
    HeterogeneousState,
    NumericRangeDomain,
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
    right = HeterogeneousState(
        (False, 0.2),
        (BooleanDomain(), NumericRangeDomain()),
    )
    with pytest.raises(ValueError):
        left.join(right)


def test_numeric_domain_rejects_nan_and_out_of_range_values():
    domain = NumericRangeDomain()
    with pytest.raises(ValueError):
        domain.validate(float("nan"))
    with pytest.raises(ValueError):
        domain.validate(1.1)
