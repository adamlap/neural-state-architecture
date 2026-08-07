"""
tests/test_nsa.py
=================
Unit test suite for Neural State Architecture (NSA) components.
"""

import unittest

from nsa.algebra import (
    StateLabel,
    StateLattice,
    ConservationLaw,
    DEFAULT_LATTICE,
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestStateAlgebra(unittest.TestCase):
    """Test suite for state algebra, lattice operations, and conservation laws."""

    def test_state_label_ordering(self):
        """Verify state label numerical hierarchy."""
        self.assertTrue(StateLabel.SYSTEM > StateLabel.PRIVATE)
        self.assertTrue(StateLabel.PRIVATE > StateLabel.CONFIDENTIAL)
        self.assertTrue(StateLabel.CONFIDENTIAL > StateLabel.TRUSTED)
        self.assertTrue(StateLabel.TRUSTED > StateLabel.PUBLIC)
        self.assertTrue(StateLabel.PUBLIC > StateLabel.UNTRUSTED)

    def test_lattice_meet_and_join(self):
        """Verify lattice meet (infimum) and join (supremum) operations."""
        lattice = DEFAULT_LATTICE
        # Meet (greatest lower bound / greatest permission overlap)
        self.assertEqual(lattice.meet(StateLabel.PRIVATE, StateLabel.PUBLIC), StateLabel.PUBLIC)
        # Join (least upper bound / least restrictive upper bound)
        self.assertEqual(lattice.join(StateLabel.PUBLIC, StateLabel.PRIVATE), StateLabel.PRIVATE)

    def test_conservation_law_monotone(self):
        """Verify monotone conservation laws in state lattice."""
        lattice = DEFAULT_LATTICE
        # Monotone transition (equal or higher sensitivity level / upward)
        self.assertTrue(lattice.is_allowed(StateLabel.PUBLIC, StateLabel.PRIVATE))
        self.assertTrue(lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PRIVATE))
        # Forbidden transition (declassification downward without explicit gate)
        self.assertFalse(lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PUBLIC))

    def test_custom_law_override(self):
        """Verify explicit conservation law creation and violation check."""
        law = ConservationLaw(
            from_label=StateLabel.PRIVATE,
            to_label=StateLabel.PUBLIC,
            allowed=False,
            penalty_weight=2.5
        )
        self.assertTrue(law.is_violated(StateLabel.PRIVATE, StateLabel.PUBLIC))
        self.assertFalse(law.is_violated(StateLabel.PUBLIC, StateLabel.PRIVATE))

    def test_product_state_vector(self):
        """Verify Product State Vector component-wise algebra."""
        from nsa.algebra import ProductStateVector
        sv1 = ProductStateVector(security=StateLabel.PUBLIC, confidence=0.9, provenance=1, license_tier=1)
        sv2 = ProductStateVector(security=StateLabel.PRIVATE, confidence=0.7, provenance=2, license_tier=2)

        joined = sv1.join_product(sv2)
        self.assertEqual(joined.security, StateLabel.PRIVATE)
        self.assertAlmostEqual(joined.confidence, 0.7)
        self.assertEqual(joined.provenance, 3)  # bitwise OR: 1 | 2 = 3
        self.assertEqual(joined.license_tier, 2)

        # Query with license_tier 2 can attend to key with license_tier 1
        self.assertTrue(sv1.allows_attention_from(sv2))
        # Query with license_tier 1 cannot attend to key with license_tier 2
        self.assertFalse(sv2.allows_attention_from(sv1))

    def test_product_lattice(self):
        """Verify Product Lattice compatibility mask generation."""
        from nsa.algebra import ProductStateVector, ProductLattice
        lattice = ProductLattice()
        q1 = ProductStateVector(security=StateLabel.SYSTEM, license_tier=3)
        k1 = ProductStateVector(security=StateLabel.UNTRUSTED, license_tier=0)
        mask = lattice.compute_mask([q1], [k1])
        self.assertEqual(mask[0][0], 0.0)  # SYSTEM query can attend to UNTRUSTED key


@unittest.skipUnless(HAS_TORCH, "PyTorch required for neural state primitives tests")
class TestStatePrimitives(unittest.TestCase):
    """Test suite for state vectors and transition operators."""

    def test_state_vector_creation(self):
        """Verify state vector initialization."""
        from nsa.state import StateVector
        sv = StateVector(state_dim=8, mode="discrete", init_label=StateLabel.PRIVATE)
        self.assertEqual(sv.state_dim, 8)
        self.assertEqual(sv.most_likely_label(), StateLabel.PRIVATE)

    def test_transition_operator(self):
        """Verify state transition operator matrix dimensions."""
        from nsa.state import StateTransitionOperator
        op = StateTransitionOperator(state_dim=8)
        self.assertEqual(op.state_dim, 8)
        self.assertEqual(op.V.shape, (8, 8))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for utility function tests")
class TestUtils(unittest.TestCase):
    """Test suite for utility functions."""

    def test_count_parameters(self):
        """Test parameter counting utility."""
        from nsa.utils import count_parameters

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)  # 100 weights + 10 biases = 110

        model = DummyModel()
        counts = count_parameters(model)
        self.assertEqual(counts["total"], 110)
        self.assertEqual(counts["trainable"], 110)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for causal LM tests")
class TestNSACausalLM(unittest.TestCase):
    """Test suite for NSACausalLM forward pass and logits shape."""

    def test_causal_lm_forward(self):
        """Verify NSACausalLM forward output tensor shapes."""
        from nsa.layers import NSACausalLM
        model = NSACausalLM(vocab_size=100, d_model=32, state_dim=4, num_layers=2, num_heads=4)
        tokens = torch.randint(0, 100, (2, 16))
        states = torch.randn(2, 16, 4)

        logits, x_out, state_out = model(tokens, state_init=states)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertEqual(x_out.shape, (2, 16, 32))
        self.assertEqual(state_out.shape, (2, 16, 4))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for fused attention tests")
class TestFusedStateAwareAttention(unittest.TestCase):
    """Test suite for FusedStateAwareAttention forward pass and shapes."""

    def test_fused_attention_forward(self):
        """Verify FusedStateAwareAttention output shape and state preservation."""
        from nsa.fused_attention import FusedStateAwareAttention
        attn = FusedStateAwareAttention(d_model=32, state_dim=4, num_heads=4, gate_mode="soft")
        x = torch.randn(2, 16, 32)
        state = torch.randn(2, 16, 4)
        mask = torch.tril(torch.ones(16, 16)).unsqueeze(0).unsqueeze(0)

        out, state_out = attn(x, state, mask=mask)
        self.assertEqual(out.shape, (2, 16, 32))
        self.assertEqual(state_out.shape, (2, 16, 4))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for LoRA tests")
class TestNSALoRA(unittest.TestCase):
    """Test suite for NSA-LoRA linear adapters and retrofitting."""

    def test_lora_linear_forward(self):
        """Verify NSALoRALinear output shape and parameter freezing."""
        from nsa.lora import NSALoRALinear
        base_layer = nn.Linear(32, 64)
        adapter = NSALoRALinear(base_layer, r=4)

        # Base layer parameters must be frozen
        for p in adapter.base_layer.parameters():
            self.assertFalse(p.requires_grad)

        # Adapter parameters must be trainable
        self.assertTrue(adapter.lora_A.requires_grad)
        self.assertTrue(adapter.lora_B.requires_grad)

        x = torch.randn(2, 16, 32)
        out = adapter(x)
        self.assertEqual(out.shape, (2, 16, 64))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for Ecosystem tests")
class TestNSAEcosystem(unittest.TestCase):
    """Test suite for Triton kernel, HuggingFace integration, and KV-cache."""

    def test_triton_fused_attention_module(self):
        """Verify FusedTritonStateAttention forward pass and shapes."""
        from nsa.triton_kernel import FusedTritonStateAttention
        attn = FusedTritonStateAttention(d_model=32, state_dim=4, num_heads=4)
        x = torch.randn(2, 16, 32)
        state = torch.randn(2, 16, 4)
        out, state_out = attn(x, state)
        self.assertEqual(out.shape, (2, 16, 32))
        self.assertEqual(state_out.shape, (2, 16, 4))

    def test_hf_causal_lm_integration(self):
        """Verify NSAForCausalLM HuggingFace interface output dict."""
        from nsa.hf_integration import NSAConfig, NSAForCausalLM
        config = NSAConfig(vocab_size=64, d_model=32, state_dim=4, num_layers=2, num_heads=4)
        hf_model = NSAForCausalLM(config)
        input_ids = torch.randint(0, 64, (2, 16))
        labels = input_ids.clone()
        res = hf_model(input_ids, labels=labels)
        self.assertIn("loss", res)
        self.assertIn("logits", res)
        self.assertEqual(res["logits"].shape, (2, 16, 64))

    def test_kv_cache_state_tracking(self):
        """Verify NSAKVCache update and state tracking behavior."""
        from nsa.kv_cache import NSAKVCache
        cache = NSAKVCache(batch_size=2, max_seq_len=32, num_heads=4, d_head=8, state_dim=4)
        k_new = torch.randn(2, 4, 4, 8)
        v_new = torch.randn(2, 4, 4, 8)
        state_new = torch.randn(2, 4, 4)

        k_cached, v_cached, s_cached = cache.update(k_new, v_new, state_new)
        self.assertEqual(k_cached.shape, (2, 4, 4, 8))
        self.assertEqual(s_cached.shape, (2, 4, 4))


if __name__ == "__main__":
    unittest.main()
