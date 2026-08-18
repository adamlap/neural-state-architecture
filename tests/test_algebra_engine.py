"""Property-level tests for the heterogeneous NSA algebra."""

import pytest

from nsa.algebra_engine import (
    BooleanSetAlgebra,
    ProbabilityAlgebra,
    ProductAlgebra,
    ScalarRiskAlgebra,
    TransitionDecision,
    legal_product_transition,
)


def test_product_join_preserves_dimension_specific_semantics():
    a = ProductAlgebra({
        "permissions": BooleanSetAlgebra(frozenset({"read"})),
        "confidence": ProbabilityAlgebra(0.9),
        "risk": ScalarRiskAlgebra(0.2),
    })
    b = ProductAlgebra({
        "permissions": BooleanSetAlgebra(frozenset({"write"})),
        "confidence": ProbabilityAlgebra(0.6),
        "risk": ScalarRiskAlgebra(0.8),
    })

    joined = a.join(b)
    assert joined.get("permissions").values == frozenset({"read", "write"})
    assert joined.get("confidence").confidence == 0.6
    assert joined.get("risk").risk == 0.8


def test_product_order_is_componentwise():
    low = ProductAlgebra({
        "permissions": BooleanSetAlgebra(frozenset({"read"})),
        "confidence": ProbabilityAlgebra(0.9),
        "risk": ScalarRiskAlgebra(0.2),
    })
    high = ProductAlgebra({
        "permissions": BooleanSetAlgebra(frozenset({"read", "write"})),
        "confidence": ProbabilityAlgebra(0.7),
        "risk": ScalarRiskAlgebra(0.8),
    })
    assert low.leq(high)
    assert legal_product_transition(low, high) == TransitionDecision.ACCEPT


def test_product_schema_mismatch_is_rejected():
    a = ProductAlgebra({"risk": ScalarRiskAlgebra(0.1)})
    b = ProductAlgebra({"confidence": ProbabilityAlgebra(0.9)})
    with pytest.raises(TypeError):
        a.join(b)


def test_probability_domain_is_bounded():
    with pytest.raises(ValueError):
        ProbabilityAlgebra(1.1)


def test_set_algebra_meet_is_intersection():
    a = BooleanSetAlgebra(frozenset({"read", "write"}))
    b = BooleanSetAlgebra(frozenset({"read", "execute"}))
    assert a.meet(b).values == frozenset({"read"})
