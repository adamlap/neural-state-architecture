"""
tests/test_state_composability.py
=================================
Verification of Product State Orthogonality & 5x5 Cross-Product Composability Matrix:

    Theorem (5x5 Cross-Product Orthogonality):
        For any product state Sigma = Sigma_C x Sigma_I x Sigma_A x Sigma_L x Sigma_R:
            For all (X, Y) with X != Y:
                Delta X => Y' == Y
            and
                (s1 join s2)_X == (s1_X join_X s2_X)

    Matrix:
                 C   I   A   L   R
        C        ✓   ✓   ✓   ✓   ✓
        I        ✓   ✓   ✓   ✓   ✓
        A        ✓   ✓   ✓   ✓   ✓
        L        ✓   ✓   ✓   ✓   ✓
        R        ✓   ✓   ✓   ✓   ✓
"""

import unittest
from nsa.algebra import (
    ConfidentialityLabel,
    IntegrityLabel,
    ProductStateVector,
    HardStateVector,
    ProductLattice,
)


class TestStateComposability(unittest.TestCase):
    """Test suite for orthogonal product state algebra and 5x5 cross-product matrix."""

    def setUp(self):
        self.dims = ["C", "I", "A", "L", "R"]
        self.base_state = ProductStateVector(
            confidentiality=ConfidentialityLabel.CONFIDENTIAL,
            integrity=IntegrityLabel.TRUSTED,
            authorizations=frozenset(["role:developer", "read:docs"]),
            confidence=0.85,
            provenance=frozenset(["source_alpha"]),
            license_tier=2,
        )

    def _mutate_dim(self, state: ProductStateVector, dim: str) -> ProductStateVector:
        """Apply a mutation exclusively to dimension dim."""
        if dim == "C":
            return ProductStateVector(
                confidentiality=ConfidentialityLabel.SYSTEM,
                integrity=state.integrity,
                authorizations=state.authorizations,
                confidence=state.confidence,
                provenance=state.provenance,
                license_tier=state.license_tier,
            )
        elif dim == "I":
            return ProductStateVector(
                confidentiality=state.confidentiality,
                integrity=IntegrityLabel.UNTRUSTED,
                authorizations=state.authorizations,
                confidence=state.confidence,
                provenance=state.provenance,
                license_tier=state.license_tier,
            )
        elif dim == "A":
            return ProductStateVector(
                confidentiality=state.confidentiality,
                integrity=state.integrity,
                authorizations=state.authorizations | frozenset(["admin:root"]),
                confidence=state.confidence,
                provenance=state.provenance,
                license_tier=state.license_tier,
            )
        elif dim == "L":
            return ProductStateVector(
                confidentiality=state.confidentiality,
                integrity=state.integrity,
                authorizations=state.authorizations,
                confidence=state.confidence,
                provenance=state.provenance,
                license_tier=state.license_tier + 2,
            )
        elif dim == "R":
            return ProductStateVector(
                confidentiality=state.confidentiality,
                integrity=state.integrity,
                authorizations=state.authorizations,
                confidence=0.30,  # Elevated risk / lower confidence
                provenance=state.provenance,
                license_tier=state.license_tier,
            )
        raise ValueError(f"Unknown dimension: {dim}")

    def _get_dim_val(self, state: ProductStateVector, dim: str):
        if dim == "C":
            return state.confidentiality
        elif dim == "I":
            return state.integrity
        elif dim == "A":
            return state.authorizations
        elif dim == "L":
            return state.license_tier
        elif dim == "R":
            return state.confidence
        raise ValueError(f"Unknown dimension: {dim}")

    def test_5x5_cross_product_orthogonality_matrix(self):
        """Exhaustively verify all 25 pairwise cross-product interactions."""
        matrix_results = {}

        for x in self.dims:
            mutated = self._mutate_dim(self.base_state, x)
            for y in self.dims:
                if x == y:
                    # Diagonal: dimension X must have changed
                    changed = self._get_dim_val(mutated, y) != self._get_dim_val(self.base_state, y)
                    self.assertTrue(
                        changed,
                        f"Diagonal ({x}, {y}) failed: mutating {x} did not update {y}!",
                    )
                    matrix_results[(x, y)] = True
                else:
                    # Off-diagonal: mutating X must NOT change dimension Y
                    unchanged = self._get_dim_val(mutated, y) == self._get_dim_val(self.base_state, y)
                    self.assertTrue(
                        unchanged,
                        f"Off-diagonal ({x}, {y}) failed: mutating {x} leaked into {y}!",
                    )
                    matrix_results[(x, y)] = True

        # Ensure all 25 permutations passed
        self.assertEqual(len(matrix_results), 25)
        for k, v in matrix_results.items():
            self.assertTrue(v, f"Pair {k} failed orthogonality check!")

    def test_product_lattice_join_and_meet_composition(self):
        """Verify (sigma_1 join sigma_2)_dim == sigma_{1, dim} join_dim sigma_{2, dim} for all dimensions."""
        s1 = ProductStateVector(
            confidentiality=ConfidentialityLabel.PUBLIC,
            integrity=IntegrityLabel.TRUSTED,
            authorizations=frozenset(["role:user"]),
            confidence=0.9,
            provenance=frozenset(["source_a"]),
            license_tier=1,
        )
        s2 = ProductStateVector(
            confidentiality=ConfidentialityLabel.PRIVATE,
            integrity=IntegrityLabel.UNTRUSTED,
            authorizations=frozenset(["role:analyst", "role:user"]),
            confidence=0.7,
            provenance=frozenset(["source_b"]),
            license_tier=3,
        )

        # Product Join
        joined = s1.join_product(s2)
        self.assertEqual(joined.confidentiality, ConfidentialityLabel.PRIVATE)
        self.assertEqual(joined.integrity, IntegrityLabel.UNTRUSTED)
        self.assertEqual(joined.authorizations, frozenset(["role:user", "role:analyst"]))
        self.assertEqual(joined.confidence, 0.7)  # Lowest confidence
        self.assertEqual(joined.provenance, frozenset(["source_a", "source_b"]))
        self.assertEqual(joined.license_tier, 3)

        # Product Meet
        met = s1.meet_product(s2)
        self.assertEqual(met.confidentiality, ConfidentialityLabel.PUBLIC)
        self.assertEqual(met.integrity, IntegrityLabel.TRUSTED)
        self.assertEqual(met.authorizations, frozenset(["role:user"]))
        self.assertEqual(met.confidence, 0.9)  # Highest confidence
        self.assertEqual(met.provenance, frozenset())  # Empty intersection
        self.assertEqual(met.license_tier, 1)

    def test_multidimensional_attention_gating_independence(self):
        """Verify that a policy violation in ANY single dimension blocks attention independently."""
        # Query with high confidentiality (SYSTEM), but low integrity and license tier
        q = ProductStateVector(
            confidentiality=ConfidentialityLabel.SYSTEM,
            integrity=IntegrityLabel.TRUSTED,
            license_tier=1,
        )

        # Key with low confidentiality (PUBLIC) -> OK, but higher license tier (3) -> BLOCKED
        k_bad_license = ProductStateVector(
            confidentiality=ConfidentialityLabel.PUBLIC,
            integrity=IntegrityLabel.TRUSTED,
            license_tier=3,
        )
        self.assertFalse(k_bad_license.allows_attention_from(q))

        # Key with higher integrity taint (UNTRUSTED = 1) -> query with TRUSTED (0) is blocked
        q_clean = ProductStateVector(
            confidentiality=ConfidentialityLabel.SYSTEM,
            integrity=IntegrityLabel.TRUSTED,
            license_tier=3,
        )
        k_tainted = ProductStateVector(
            confidentiality=ConfidentialityLabel.PUBLIC,
            integrity=IntegrityLabel.UNTRUSTED,
            license_tier=1,
        )
        self.assertFalse(k_tainted.allows_attention_from(q_clean))

        # Key with all dimensions satisfied -> ALLOWED
        k_valid = ProductStateVector(
            confidentiality=ConfidentialityLabel.PUBLIC,
            integrity=IntegrityLabel.TRUSTED,
            license_tier=1,
        )
        self.assertTrue(k_valid.allows_attention_from(q_clean))


if __name__ == "__main__":
    unittest.main()
