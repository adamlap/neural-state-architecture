"""
Property-based tests for NSA algebra.
"""

import pytest
from hypothesis import given, strategies as st
from typing import FrozenSet

from nsa.algebra import (
    ConfidentialityLabel,
    IntegrityLabel,
    ProductStateVector,
    DEFAULT_LATTICE,
    INTEGRITY_LATTICE,
    ProductLattice,
)

# Strategies
st_confidentiality = st.sampled_from(list(ConfidentialityLabel))
st_integrity = st.sampled_from(list(IntegrityLabel))
st_confidence = st.floats(min_value=0.0, max_value=1.0)
st_provenance = st.frozensets(st.text())
st_license = st.integers(min_value=0, max_value=7)

st_product_state = st.builds(
    ProductStateVector,
    confidentiality=st_confidentiality,
    integrity=st_integrity,
    confidence=st_confidence,
    provenance=st_provenance,
    license_tier=st_license,
)

class TestConfidentialityLatticeProperties:
    @given(a=st_confidentiality, b=st_confidentiality)
    def test_commutativity(self, a: ConfidentialityLabel, b: ConfidentialityLabel):
        assert DEFAULT_LATTICE.join(a, b) == DEFAULT_LATTICE.join(b, a)
        assert DEFAULT_LATTICE.meet(a, b) == DEFAULT_LATTICE.meet(b, a)

    @given(a=st_confidentiality, b=st_confidentiality, c=st_confidentiality)
    def test_associativity(self, a: ConfidentialityLabel, b: ConfidentialityLabel, c: ConfidentialityLabel):
        assert DEFAULT_LATTICE.join(a, DEFAULT_LATTICE.join(b, c)) == DEFAULT_LATTICE.join(DEFAULT_LATTICE.join(a, b), c)
        assert DEFAULT_LATTICE.meet(a, DEFAULT_LATTICE.meet(b, c)) == DEFAULT_LATTICE.meet(DEFAULT_LATTICE.meet(a, b), c)

    @given(a=st_confidentiality)
    def test_idempotence(self, a: ConfidentialityLabel):
        assert DEFAULT_LATTICE.join(a, a) == a
        assert DEFAULT_LATTICE.meet(a, a) == a

    @given(a=st_confidentiality, b=st_confidentiality)
    def test_absorption(self, a: ConfidentialityLabel, b: ConfidentialityLabel):
        assert DEFAULT_LATTICE.join(a, DEFAULT_LATTICE.meet(a, b)) == a
        assert DEFAULT_LATTICE.meet(a, DEFAULT_LATTICE.join(a, b)) == a

class TestProductLatticeProperties:
    @given(a=st_product_state, b=st_product_state)
    def test_commutativity(self, a: ProductStateVector, b: ProductStateVector):
        assert a.join_product(b) == b.join_product(a)
        assert a.meet_product(b) == b.meet_product(a)

    @given(a=st_product_state, b=st_product_state, c=st_product_state)
    def test_associativity(self, a: ProductStateVector, b: ProductStateVector, c: ProductStateVector):
        assert a.join_product(b.join_product(c)) == (a.join_product(b)).join_product(c)
        assert a.meet_product(b.meet_product(c)) == (a.meet_product(b)).meet_product(c)

    @given(a=st_product_state)
    def test_idempotence(self, a: ProductStateVector):
        assert a.join_product(a) == a
        assert a.meet_product(a) == a

    @given(a=st_product_state, b=st_product_state)
    def test_absorption(self, a: ProductStateVector, b: ProductStateVector):
        assert a.join_product(a.meet_product(b)) == a
        assert a.meet_product(a.join_product(b)) == a
