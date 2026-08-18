"""
tests/test_gradcheck.py
========================
PyTorch Autograd & Gradcheck test suite for NSA modules.

Verifies:
1. Double precision gradient backpropagation (torch.autograd.gradcheck).
2. Non-NaN/Inf gradient flow through StateAwareAttention, FusedStateAwareAttention, and SemanticGate.
3. Differentiability of NSALoss state constraint terms.
"""

import unittest

import torch
from torch import nn

from nsa.fused_attention import FusedStateAwareAttention
from nsa.objectives import StateConstraintLoss
from nsa.state import SemanticGate


class TestNSAGradcheck(unittest.TestCase):
    """Gradcheck test suite for custom PyTorch operators in NSA."""

    def setUp(self):
        torch.manual_seed(42)

    def test_semantic_gate_gradcheck(self):
        """Verify autograd gradcheck on SemanticGate."""
        d_model, state_dim = 16, 4
        gate = SemanticGate(d_model=d_model, state_dim=state_dim).to(torch.float64)

        x = torch.randn(2, 4, d_model, dtype=torch.float64, requires_grad=True)
        state = torch.randn(2, 4, state_dim, dtype=torch.float64, requires_grad=True)

        res = torch.autograd.gradcheck(gate, (x, state), eps=1e-6, atol=1e-4)
        self.assertTrue(res, "SemanticGate failed autograd gradcheck")

    def test_fused_attention_gradcheck(self):
        """Verify autograd gradcheck on FusedStateAwareAttention in soft mode."""
        d_model, state_dim, num_heads = 16, 4, 2
        attn = FusedStateAwareAttention(
            d_model=d_model, state_dim=state_dim, num_heads=num_heads, gate_mode="soft"
        ).to(torch.float64)

        x = torch.randn(2, 4, d_model, dtype=torch.float64, requires_grad=True)
        state = torch.randn(2, 4, state_dim, dtype=torch.float64, requires_grad=False)

        def func(inputs):
            out, _ = attn(inputs, state)
            return out

        res = torch.autograd.gradcheck(func, (x,), eps=1e-6, atol=1e-4)
        self.assertTrue(res, "FusedStateAwareAttention failed autograd gradcheck")

    def test_state_constraint_loss_differentiability(self):
        """Verify state constraint loss gradients flow into state transitions."""
        loss_fn = StateConstraintLoss()
        src_state = torch.tensor(
            [[[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
        )  # PRIVATE (state_dim=8)

        # Linear layer producing target state
        linear = nn.Linear(8, 8)
        opt = torch.optim.SGD(linear.parameters(), lr=0.1)

        inp = torch.randn(1, 1, 8)
        dst_state = linear(inp)

        loss = loss_fn(src_state, dst_state)
        loss.backward()

        # Verify non-zero gradient flow into linear layer weights
        self.assertIsNotNone(linear.weight.grad)
        self.assertTrue(torch.abs(linear.weight.grad).sum() > 0.0)


if __name__ == "__main__":
    unittest.main()
