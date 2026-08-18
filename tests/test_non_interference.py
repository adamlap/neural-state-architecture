"""
tests/test_non_interference.py
==============================
Rigorous mathematical tests for Local & Compositional Observational Equivalence Non-Interference:

    Theorem (Structural Non-Interference):
        For any observer clearance L in Sigma_h:
            X =_L X' ==> Obs_L(F(X)) == Obs_L(F(X'))

Where:
    - X =_L X' (Low-Equivalence): for all t, sigma_{h, t} <= L ==> X_t == X'_t
    - Obs_L(Y): projections and output representations at positions where sigma_{h, t} <= L.
"""

import unittest

import torch
from torch import nn

from nsa.algebra import StateLabel
from nsa.attention import StateAwareAttention
from nsa.layers import GatedFFN, NSATransformerBlock
from nsa.types import TypedTensor
from nsa.utils import state_labels_to_vectors


class TestNonInterference(unittest.TestCase):
    """Rigorous observational equivalence and local operator invariant testing."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 32
        self.state_dim = 8
        self.num_heads = 4

    def test_local_attention_non_interference(self):
        """Test Local Operator 1: StateAwareAttention preserves low-equivalence."""
        attn = StateAwareAttention(
            d_model=self.d_model,
            state_dim=self.state_dim,
            num_heads=self.num_heads,
            compat_mode="level",
            gate_mode="hard",
            use_discrete_levels=True,
        )
        attn.eval()

        # Sequence: [PUBLIC, SYSTEM, PUBLIC]
        labels = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.PUBLIC.value]])
        state = state_labels_to_vectors(labels, state_dim=self.state_dim, noise=0.0)

        # Base input X
        x_base = torch.randn(1, 3, self.d_model)
        # Perturbed input X': SYSTEM token at index 1 is altered completely; PUBLIC tokens 0 and 2 are identical
        x_pert = x_base.clone()
        x_pert[0, 1, :] += 50.0  # Large perturbation on secret

        with torch.no_grad():
            out_base, _ = attn(x_base, state)
            out_pert, _ = attn(x_pert, state)

        # Observer at level PUBLIC: positions 0 and 2 must be IDENTICAL
        self.assertTrue(torch.allclose(out_base[0, 0, :], out_pert[0, 0, :], atol=1e-5), "Public position 0 leaked secret!")
        self.assertTrue(torch.allclose(out_base[0, 2, :], out_pert[0, 2, :], atol=1e-5), "Public position 2 leaked secret!")
        # Secret position 1 should differ
        self.assertFalse(torch.allclose(out_base[0, 1, :], out_pert[0, 1, :], atol=1e-5))

    def test_local_ffn_non_interference(self):
        """Test Local Operator 2: Position-wise GatedFFN is strictly local."""
        ffn = GatedFFN(d_model=self.d_model, state_dim=self.state_dim)
        ffn.eval()

        labels = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value]])
        state = state_labels_to_vectors(labels, state_dim=self.state_dim, noise=0.0)

        x_base = torch.randn(1, 2, self.d_model)
        x_pert = x_base.clone()
        x_pert[0, 1, :] += 10.0

        with torch.no_grad():
            out_base = ffn(x_base, state)
            out_pert = ffn(x_pert, state)

        # FFN at position 0 depends only on position 0
        self.assertTrue(torch.allclose(out_base[0, 0, :], out_pert[0, 0, :], atol=1e-5))

    def test_local_residual_addition_non_interference(self):
        """Test Local Operator 3: TypedTensor residual addition maintains product lattice invariants."""
        labels1 = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value]])
        labels2 = torch.tensor([[StateLabel.CONFIDENTIAL.value, StateLabel.PUBLIC.value]])

        state1 = state_labels_to_vectors(labels1, state_dim=self.state_dim, noise=0.0)
        state2 = state_labels_to_vectors(labels2, state_dim=self.state_dim, noise=0.0)

        t1 = TypedTensor(m=torch.randn(1, 2, self.d_model), sigma_h=state1, sigma_s=torch.ones(1, 2, 1), nu=torch.zeros(1, 2, 1))
        t2 = TypedTensor(m=torch.randn(1, 2, self.d_model), sigma_h=state2, sigma_s=torch.ones(1, 2, 1), nu=torch.zeros(1, 2, 1))

        t_join = t1.join_with(t2)
        # Position 0: max(PUBLIC=1, CONFIDENTIAL=3) = CONFIDENTIAL(3)
        self.assertEqual(t_join.sigma_h[0, 0, 0].item(), StateLabel.CONFIDENTIAL.value)
        # Position 1: max(SYSTEM=5, PUBLIC=1) = SYSTEM(5)
        self.assertEqual(t_join.sigma_h[0, 1, 0].item(), StateLabel.SYSTEM.value)

    def test_compositional_multiblock_non_interference(self):
        """Test Compositional Multi-Block Invariant: Obs_L(F(X)) == Obs_L(F(X'))."""
        num_layers = 4
        blocks = nn.ModuleList([
            NSATransformerBlock(
                d_model=self.d_model,
                state_dim=self.state_dim,
                num_heads=self.num_heads,
                gate_mode="hard",
                compat_mode="level",
            )
            for _ in range(num_layers)
        ])
        for b in blocks:
            b.eval()

        # Sequence: [PUBLIC, SYSTEM, PUBLIC, PUBLIC]
        labels = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.PUBLIC.value, StateLabel.PUBLIC.value]])
        state = state_labels_to_vectors(labels, state_dim=self.state_dim, noise=0.0)

        x_base = torch.randn(1, 4, self.d_model)
        x_pert = x_base.clone()
        x_pert[0, 1, :] = torch.randn(self.d_model) * 10.0  # Completely different secret

        t_base = TypedTensor(m=x_base, sigma_h=state, sigma_s=torch.ones(1, 4, 1), nu=torch.zeros(1, 4, 1))
        t_pert = TypedTensor(m=x_pert, sigma_h=state, sigma_s=torch.ones(1, 4, 1), nu=torch.zeros(1, 4, 1))

        with torch.no_grad():
            for block in blocks:
                t_base = block(t_base)
                t_pert = block(t_pert)

        # Verify Obs_PUBLIC(F(X)) == Obs_PUBLIC(F(X')) at all public positions (0, 2, 3)
        for pub_idx in [0, 2, 3]:
            self.assertTrue(
                torch.allclose(t_base.m[0, pub_idx, :], t_pert.m[0, pub_idx, :], atol=1e-4),
                f"Multi-block composition leaked secret to public position {pub_idx}!",
            )

        # Secret position 1 must differ
        self.assertFalse(torch.allclose(t_base.m[0, 1, :], t_pert.m[0, 1, :], atol=1e-4))

    def test_hard_state_immutability_under_continuous_layers(self):
        """Phase 9: Verify hard state cannot be mutated by un-authenticated continuous neural computation."""
        block = NSATransformerBlock(
            d_model=self.d_model,
            state_dim=self.state_dim,
            num_heads=self.num_heads,
            gate_mode="hard",
            compat_mode="level",
        )
        labels = torch.tensor([[StateLabel.UNTRUSTED.value, StateLabel.CONFIDENTIAL.value, StateLabel.PRIVATE.value, StateLabel.SYSTEM.value]])
        initial_sigma_h = state_labels_to_vectors(labels, state_dim=self.state_dim, noise=0.0)

        typed_in = TypedTensor(
            m=torch.randn(1, 4, self.d_model),
            sigma_h=initial_sigma_h,
            sigma_s=torch.rand(1, 4, 1),
            nu=torch.rand(1, 4, 1),
        )

        typed_out = block(typed_in)
        # Coordinate 0 (discrete security level) must be identically preserved
        self.assertTrue(torch.equal(typed_out.sigma_h[..., 0], initial_sigma_h[..., 0]), "Hard security coordinate was mutated by neural layer!")


if __name__ == "__main__":
    unittest.main()
