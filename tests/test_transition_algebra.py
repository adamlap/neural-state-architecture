"""
tests/test_transition_algebra.py
================================
Formal invariant and algebraic property tests for state transition operators V in T_Sigma.

Mathematical convention:
    sigma' = sigma @ V.T, or equivalently sigma'_j = sum_i sigma_i * V_{j, i}
    Row j = destination state (dst)
    Column i = source state (src)
    Legal transition dst >= src (row >= col) -> LOWER TRIANGULAR.
"""

import unittest

import torch

from nsa.algebra import StateLabel, TransitionOperator, project_transition_matrix
from nsa.state import StateTransitionOperator


class TestTransitionAlgebra(unittest.TestCase):
    """Rigorous property testing for transition matrix projection P_{T_Sigma}(V)."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_state = len(StateLabel)  # 6 levels

    def test_1_projection_legality(self):
        """Test 1: P(V) in T_Sigma for random matrices V."""
        for _ in range(20):
            V = torch.randn(self.d_state, self.d_state) * 5.0
            V_proj = project_transition_matrix(V, monotone=True)

            # Check that upper triangle (dst < src, row < col) is identically 0.0
            upper_tri = torch.triu(V_proj, diagonal=1)
            self.assertTrue(torch.all(upper_tri == 0.0), "Found non-zero forbidden transitions in upper triangle!")

    def test_2_projection_idempotence(self):
        """Test 2: P(P(V)) == P(V) (Idempotence property)."""
        for _ in range(20):
            V = torch.randn(self.d_state, self.d_state) * 10.0
            V_proj1 = project_transition_matrix(V, monotone=True)
            V_proj2 = project_transition_matrix(V_proj1, monotone=True)

            self.assertTrue(torch.allclose(V_proj1, V_proj2, atol=1e-7), "Projection is not idempotent!")

    def test_3_forbidden_transitions_zero(self):
        """Test 3: For every dst < src, P(V)[dst, src] == 0.0."""
        V = torch.ones(self.d_state, self.d_state) * 3.14
        V_proj = project_transition_matrix(V, monotone=True)

        for src in range(self.d_state):
            for dst in range(self.d_state):
                if dst < src:
                    self.assertEqual(
                        V_proj[dst, src].item(),
                        0.0,
                        f"Forbidden transition {StateLabel(src).name} -> {StateLabel(dst).name} (V[{dst}, {src}]) is non-zero!",
                    )

    def test_4_legal_transitions_preserved(self):
        """Test 4: Legal entries dst >= src are preserved according to projection semantics."""
        V = torch.tensor([
            [1.5, -2.0, 3.0],
            [4.0, -1.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        V_proj = project_transition_matrix(V, monotone=True)

        # Diagonal clamped >= 0
        self.assertEqual(V_proj[0, 0].item(), 1.5)
        self.assertEqual(V_proj[1, 1].item(), 0.0)  # max(0, -1.0)
        self.assertEqual(V_proj[2, 2].item(), 9.0)

        # Strictly lower triangle preserved
        self.assertEqual(V_proj[1, 0].item(), 4.0)
        self.assertEqual(V_proj[2, 0].item(), 7.0)
        self.assertEqual(V_proj[2, 1].item(), 8.0)

        # Strictly upper triangle zeroed
        self.assertEqual(V_proj[0, 1].item(), 0.0)
        self.assertEqual(V_proj[0, 2].item(), 0.0)
        self.assertEqual(V_proj[1, 2].item(), 0.0)

    def test_5_basis_state_transition_support(self):
        """Test 5: For every basis state e_src, output support under sigma' = e_src @ V.T exists ONLY in {dst: dst >= src}."""
        op = StateTransitionOperator(state_dim=self.d_state, monotone_clamp=True)
        # Populate with dense random weights
        with torch.no_grad():
            op.V.copy_(torch.randn(self.d_state, self.d_state).abs() + 0.5)

        for src in range(self.d_state):
            e_src = torch.zeros(1, self.d_state)
            e_src[0, src] = 1.0

            # Forward propagation: sigma' = e_src @ V.T
            sigma_next = op(e_src)

            # Check that for all dst < src, sigma_next[0, dst] == 0.0
            for dst in range(self.d_state):
                if dst < src:
                    self.assertEqual(
                        sigma_next[0, dst].item(),
                        0.0,
                        f"Basis transition from src={StateLabel(src).name} produced unauthorized support at dst={StateLabel(dst).name} (level {dst})!",
                    )
                else:
                    # Legal destination should have non-negative value
                    self.assertGreaterEqual(
                        sigma_next[0, dst].item(),
                        0.0,
                    )

    def test_6_regression_transition_operator(self):
        """Test 6: TransitionOperator in nsa/algebra.py respects lower-triangular projection."""
        op = TransitionOperator(d_state=self.d_state)
        with torch.no_grad():
            op.weight.copy_(torch.randn(self.d_state, self.d_state))

        V_proj = op.get_projected_weight()
        upper_tri = torch.triu(V_proj, diagonal=1)
        self.assertTrue(torch.all(upper_tri == 0.0))


if __name__ == "__main__":
    unittest.main()
