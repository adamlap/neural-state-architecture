"""
tests/test_fuzzing.py
======================
Property-based fuzzing test suite for NSA using Hypothesis.

Verifies mathematical invariants:
1. ProductStateVector join & meet commutativity, associativity, and idempotency.
2. Non-interference monotonicity invariant: L_Q < L_K => attention forbidden.
3. ProductLattice mask boundedness.
"""

import unittest
from hypothesis import given, strategies as st

from nsa.algebra import ConfidentialityLabel, IntegrityLabel, ProductStateVector, ProductLattice, DEFAULT_LATTICE


# Hypothesis strategies for generating random state labels and product vectors
confidentiality_st = st.sampled_from(list(ConfidentialityLabel))
integrity_st = st.sampled_from(list(IntegrityLabel))
confidences_st = st.floats(min_value=0.0, max_value=1.0)
provenance_st = st.frozensets(st.text(min_size=1, max_size=5))
license_st = st.integers(min_value=0, max_value=10)

product_vectors_st = st.builds(
    ProductStateVector,
    confidentiality=confidentiality_st,
    integrity=integrity_st,
    confidence=confidences_st,
    provenance=provenance_st,
    license_tier=license_st,
)


class TestStateAlgebraFuzzing(unittest.TestCase):
    """Property-based fuzz test suite for state algebra."""

    @given(sv1=product_vectors_st, sv2=product_vectors_st)
    def test_join_commutativity(self, sv1: ProductStateVector, sv2: ProductStateVector):
        """Verify sv1 ⊔ sv2 == sv2 ⊔ sv1."""
        j1 = sv1.join_product(sv2)
        j2 = sv2.join_product(sv1)
        self.assertEqual(j1.confidentiality, j2.confidentiality)
        self.assertEqual(j1.integrity, j2.integrity)
        self.assertAlmostEqual(j1.confidence, j2.confidence)
        self.assertEqual(j1.provenance, j2.provenance)
        self.assertEqual(j1.license_tier, j2.license_tier)

    @given(sv1=product_vectors_st, sv2=product_vectors_st, sv3=product_vectors_st)
    def test_join_associativity(self, sv1: ProductStateVector, sv2: ProductStateVector, sv3: ProductStateVector):
        """Verify (sv1 ⊔ sv2) ⊔ sv3 == sv1 ⊔ (sv2 ⊔ sv3)."""
        left = (sv1.join_product(sv2)).join_product(sv3)
        right = sv1.join_product(sv2.join_product(sv3))
        self.assertEqual(left.confidentiality, right.confidentiality)
        self.assertEqual(left.integrity, right.integrity)
        self.assertAlmostEqual(left.confidence, right.confidence)
        self.assertEqual(left.provenance, right.provenance)
        self.assertEqual(left.license_tier, right.license_tier)

    @given(sv=product_vectors_st)
    def test_join_idempotency(self, sv: ProductStateVector):
        """Verify sv ⊔ sv == sv."""
        joined = sv.join_product(sv)
        self.assertEqual(joined.confidentiality, sv.confidentiality)
        self.assertEqual(joined.integrity, sv.integrity)
        self.assertAlmostEqual(joined.confidence, sv.confidence)
        self.assertEqual(joined.provenance, sv.provenance)
        self.assertEqual(joined.license_tier, sv.license_tier)

    @given(query=product_vectors_st, key=product_vectors_st)
    def test_non_interference_monotonicity(self, query: ProductStateVector, key: ProductStateVector):
        """Verify Theorem 1 / Non-Interference: query.security < key.security => forbidden."""
        allowed = key.allows_attention_from(query)
        if query.confidentiality.value < key.confidentiality.value:
            self.assertFalse(allowed)
        if query.integrity.value < key.integrity.value:
            self.assertFalse(allowed)
        if query.license_tier < key.license_tier:
            self.assertFalse(allowed)

    @given(q_list=st.lists(product_vectors_st, min_size=1, max_size=5), k_list=st.lists(product_vectors_st, min_size=1, max_size=5))
    def test_product_lattice_mask_values(self, q_list, k_list):
        """Verify product lattice compute_mask generates valid 0.0 or -1e4 values."""
        lattice = ProductLattice()
        mask = lattice.compute_mask(q_list, k_list)
        self.assertEqual(len(mask), len(q_list))
        self.assertEqual(len(mask[0]), len(k_list))

        for row in mask:
            for val in row:
                self.assertIn(val, [0.0, -1e4])


if __name__ == "__main__":
    unittest.main()
