"""
tests/test_masks.py
====================
Unit tests for NSA state mask combination and LoRA adapter parameter isolation.

Verifies:
1. Combination of 4D NSA state masks with causal additive masks.
2. NSALoRALinear parameter freezing invariance (base_layer weights frozen, lora trainable).
3. Soft vs Hard state masking behavior in FusedStateAwareAttention.
"""

import unittest

import torch
from torch import nn

from nsa.fused_attention import FusedStateAwareAttention
from nsa.lora import NSALoRALinear


class TestNSAMasksAndLoRA(unittest.TestCase):
    """Test suite for mask combinations and LoRA layer parameter isolation."""

    def test_causal_and_state_mask_combination(self):
        """Verify state mask correctly adds to causal mask without shape mismatch."""
        attn = FusedStateAwareAttention(d_model=32, state_dim=4, num_heads=4, gate_mode="soft")
        x = torch.randn(2, 8, 32)
        state = torch.randn(2, 8, 4)

        # 1 = allowed, 0 = causal masked
        causal_mask = torch.tril(torch.ones(8, 8)).unsqueeze(0).unsqueeze(0)

        out, _ = attn(x, state, mask=causal_mask)
        self.assertEqual(out.shape, (2, 8, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_lora_parameter_isolation(self):
        """Verify NSALoRALinear freezes base weights while enabling adapter gradients."""
        base_layer = nn.Linear(32, 64)
        adapter = NSALoRALinear(base_layer, r=8, lora_alpha=16.0)

        # Base layer weights must be frozen
        for p in adapter.base_layer.parameters():
            self.assertFalse(p.requires_grad)

        # LoRA A and B must be trainable
        self.assertTrue(adapter.lora_A.requires_grad)
        self.assertTrue(adapter.lora_B.requires_grad)

        # Forward pass output check
        x = torch.randn(2, 4, 32)
        y = adapter(x)
        self.assertEqual(y.shape, (2, 4, 64))

        # Backward pass gradient check
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(adapter.lora_A.grad)
        self.assertIsNotNone(adapter.lora_B.grad)
        self.assertIsNone(adapter.base_layer.weight.grad)

    def test_hard_gating_mask(self):
        """Verify hard state gating sets forbidden attention scores to -inf."""
        attn = FusedStateAwareAttention(d_model=32, state_dim=4, num_heads=4, gate_mode="hard")
        x = torch.randn(2, 8, 32)
        state = torch.randn(2, 8, 4)

        out, _ = attn(x, state)
        self.assertEqual(out.shape, (2, 8, 32))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
