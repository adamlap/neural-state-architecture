from hypothesis import given, strategies as st

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    CapabilityDomain,
    HeterogeneousState,
    NumericRangeDomain,
    ProbabilityInterval,
    ProbabilityIntervalDomain,
    TemporalWindow,
    TemporalWindowDomain,
)


floats01 = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
probabilities = st.builds(
    lambda a, b: ProbabilityInterval(min(a, b), max(a, b)),
    floats01,
    floats01,
)
times = st.floats(min_value=-1_000_000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
temporal_windows = st.builds(
    lambda a, b: TemporalWindow(min(a, b), max(a, b)),
    times,
    times,
)
capabilities = st.frozensets(st.sampled_from(["read", "write", "execute", "network"]))


def assert_lattice_laws(domain, a, b, c):
    join = domain.join
    meet = domain.meet
    assert join(a, b) == join(b, a)
    assert meet(a, b) == meet(b, a)
    assert join(join(a, b), c) == join(a, join(b, c))
    assert meet(meet(a, b), c) == meet(a, meet(b, c))
    assert join(a, a) == a
    assert meet(a, a) == a
    assert meet(a, join(a, b)) == a
    assert join(a, meet(a, b)) == a


@given(st.booleans(), st.booleans(), st.booleans())
def test_boolean_lattice_laws(a, b, c):
    assert_lattice_laws(BooleanDomain(), a, b, c)


@given(capabilities, capabilities, capabilities)
def test_capability_lattice_laws(a, b, c):
    assert_lattice_laws(CapabilityDomain(), a, b, c)


@given(floats01, floats01, floats01)
def test_numeric_lattice_laws(a, b, c):
    assert_lattice_laws(NumericRangeDomain(), a, b, c)


@given(probabilities, probabilities, probabilities)
def test_probability_interval_lattice_laws(a, b, c):
    assert_lattice_laws(ProbabilityIntervalDomain(), a, b, c)


@given(temporal_windows, temporal_windows, temporal_windows)
def test_temporal_window_lattice_laws(a, b, c):
    assert_lattice_laws(TemporalWindowDomain(), a, b, c)


@given(st.booleans(), floats01, capabilities, st.booleans(), floats01, capabilities)
def test_heterogeneous_product_join_is_associative_and_commutative(
    flag_a, risk_a, caps_a, flag_b, risk_b, caps_b
):
    # A product lattice inherits these laws coordinate-wise.
    domains = (BooleanDomain(), NumericRangeDomain(), CapabilityDomain())
    a = HeterogeneousState((flag_a, risk_a, caps_a), domains)
    b = HeterogeneousState((flag_b, risk_b, caps_b), domains)
    c = HeterogeneousState((False, 0.5, frozenset()), domains)
    assert a.join(b) == b.join(a)
    assert a.join(b).join(c) == a.join(b.join(c))
    assert a.meet(b) == b.meet(a)
    assert a.meet(b).meet(c) == a.meet(b.meet(c))
