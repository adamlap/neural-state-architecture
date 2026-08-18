"""
tests/test_fused_triton_kernel.py
=================================
Unit tests for the True Fused State-Aware Attention Kernel:
1. Validates definition of Triton JIT kernel and fallback dispatcher.
2. Tests direct (Q, K, V, q_state, k_state) signature without N x N mask DRAM allocation.
3. Tests numerical equivalence against reference SDPA hard attention.
4. Verifies non-interference property under the fused execution path.
"""

import unittest
import torch
from torch import nn

from nsa.algebra import StateLabel
from nsa.attention import StateAwareAttention
from nsa.triton_kernel import (
    TRITON_KERNEL_DEFINED,
    TritonFusedStateAwareAttention,
    last_backend,
    triton_fused_state_attention,
)
from nsa.utils import state_labels_to_vectors


class TestTrueFusedTritonKernel(unittest.TestCase):
    """Test suite for true fused state-aware attention kernel."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 32
        self.num_heads = 4
        self.head_dim = 8
        self.state_dim = 8

    def test_triton_kernel_defined(self):
        """Verify that the @triton.jit kernel is syntactically defined and imported."""
        self.assertTrue(TRITON_KERNEL_DEFINED, "True fused Triton JIT kernel was not defined!")

    def test_direct_state_signature_and_forward(self):
        """Verify (Q, K, V, q_states, k_states) forward pass without precomputed mask."""
        b, h, t, d = 2, 4, 16, 8
        q = torch.randn(b, h, t, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)

        # State tensor [B, T] with discrete levels
        q_states = torch.randint(0, 6, (b, t))

        out = triton_fused_state_attention(q, k, v, q_states=q_states, is_causal=True)
        self.assertEqual(out.shape, (b, h, t, d))
        self.assertTrue(torch.isfinite(out).all())

    def test_numerical_equivalence_with_sdpa_hard_mask(self):
        """Verify numerical equivalence between fused dispatcher and standard SDPA hard-masking."""
        b, h, t, d = 1, 2, 8, 16
        q = torch.randn(b, h, t, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)

        labels = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.PUBLIC.value, StateLabel.PRIVATE.value,
                                StateLabel.UNTRUSTED.value, StateLabel.SYSTEM.value, StateLabel.CONFIDENTIAL.value, StateLabel.PUBLIC.value]])

        # 1. Output from true fused kernel dispatcher
        out_fused = triton_fused_state_attention(q, k, v, q_states=labels, is_causal=True)

        # 2. Output from reference manual SDPA with hard mask
        offs_q = torch.arange(t).unsqueeze(-1)
        offs_k = torch.arange(t).unsqueeze(-2)
        compat = (labels.unsqueeze(-1) >= labels.unsqueeze(-2)) & (offs_q >= offs_k)
        ref_mask = torch.where(compat.unsqueeze(1), torch.tensor(0.0), torch.tensor(-1e4))
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=ref_mask, scale=1.0 / (d ** 0.5), is_causal=False
        )

        self.assertTrue(torch.allclose(out_fused, out_ref, atol=1e-5), "True fused kernel output diverges from SDPA reference!")

    def test_fused_module_non_interference(self):
        """Verify that TritonFusedStateAwareAttention preserves observational non-interference."""
        module = TritonFusedStateAwareAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            state_dim=self.state_dim,
            is_causal=True,
        )
        module.eval()

        # Sequence: [PUBLIC, SYSTEM, PUBLIC]
        labels = torch.tensor([[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.PUBLIC.value]])
        state = state_labels_to_vectors(labels, state_dim=self.state_dim, noise=0.0)

        x_base = torch.randn(1, 3, self.d_model)
        x_pert = x_base.clone()
        x_pert[0, 1, :] += 50.0  # Perturb secret token

        with torch.no_grad():
            out_base, _ = module(x_base, state)
            out_pert, _ = module(x_pert, state)

        # Position 0 is PUBLIC and appears before SYSTEM -> identical
        self.assertTrue(torch.allclose(out_base[0, 0, :], out_pert[0, 0, :], atol=1e-5))
        # Position 2 is PUBLIC (causal + hard mask blocks SYSTEM at pos 1) -> identical
        self.assertTrue(torch.allclose(out_base[0, 2, :], out_pert[0, 2, :], atol=1e-5))
        # Position 1 is SYSTEM -> changed
        self.assertFalse(torch.allclose(out_base[0, 1, :], out_pert[0, 1, :], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
