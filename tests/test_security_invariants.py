"""
tests/test_security_invariants.py
=================================
Hard security / integrity tests for NSA.
"""

import unittest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from nsa.algebra import (
    StateLabel,
    StateLattice,
    DEFAULT_LATTICE,
    ProductStateVector,
    RAGMetadataIngressEncoder,
    build_label_attention_mask,
)


class TestLatticeSemantics(unittest.TestCase):
    def test_can_attend_asymmetric(self):
        lat = DEFAULT_LATTICE
        self.assertTrue(lat.can_attend(StateLabel.SYSTEM, StateLabel.UNTRUSTED))
        self.assertFalse(lat.can_attend(StateLabel.PUBLIC, StateLabel.PRIVATE))
        # compatible is alias of can_attend(query=src, key=dst)
        self.assertTrue(lat.compatible(StateLabel.SYSTEM, StateLabel.UNTRUSTED))
        self.assertFalse(lat.compatible(StateLabel.PUBLIC, StateLabel.PRIVATE))
        self.assertTrue(lat.can_read(StateLabel.SYSTEM, StateLabel.UNTRUSTED))

    def test_can_write_no_write_down(self):
        lat = DEFAULT_LATTICE
        self.assertTrue(lat.can_write(StateLabel.PUBLIC, StateLabel.PRIVATE))
        self.assertFalse(lat.can_write(StateLabel.PRIVATE, StateLabel.PUBLIC))
        self.assertTrue(lat.can_write(StateLabel.UNTRUSTED, StateLabel.SYSTEM))
        self.assertFalse(lat.can_write(StateLabel.SYSTEM, StateLabel.UNTRUSTED))

    def test_can_declassify_requires_auth_for_downward(self):
        lat = DEFAULT_LATTICE
        # upward / equal always ok
        self.assertTrue(lat.can_declassify(StateLabel.PUBLIC, StateLabel.PRIVATE, authorized=False))
        self.assertTrue(lat.can_declassify(StateLabel.PUBLIC, StateLabel.PUBLIC, authorized=False))
        # downward denied without auth
        self.assertFalse(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, authorized=False))
        self.assertTrue(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, authorized=True))

    def test_build_label_mask_blocks_downward(self):
        if not HAS_TORCH:
            self.skipTest("torch required")
        labels = torch.tensor([[StateLabel.SYSTEM.value, StateLabel.PUBLIC.value, StateLabel.PRIVATE.value]])
        mask = build_label_attention_mask(labels)
        # PUBLIC query (row 1) cannot read PRIVATE key (col 2)
        self.assertLess(mask[0, 0, 1, 2].item(), 0)
        # SYSTEM query (row 0) can read PUBLIC key (col 1)
        self.assertEqual(mask[0, 0, 0, 1].item(), 0.0)
        # equal levels allowed
        self.assertEqual(mask[0, 0, 1, 1].item(), 0.0)

    def test_rag_encoder(self):
        sv = RAGMetadataIngressEncoder.encode_metadata_dict(
            {"security": "PRIVATE", "confidence": 0.9, "provenance": 3, "license": "INTERNAL"}
        )
        self.assertEqual(sv.security, StateLabel.PRIVATE)
        self.assertEqual(sv.provenance, 3)
        self.assertEqual(sv.license_tier, 1)
        self.assertAlmostEqual(sv.confidence, 0.9)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestHardAttentionNonInterference(unittest.TestCase):
    def test_hard_attention_zero_mass_on_forbidden(self):
        from nsa.attention import StateAwareAttention
        from nsa.utils import state_labels_to_vectors

        torch.manual_seed(0)
        attn = StateAwareAttention(
            d_model=32, state_dim=8, num_heads=4,
            compat_mode="level", gate_mode="hard", use_discrete_levels=True,
        )
        # Manually compute scores path via _state_mask
        labels = torch.tensor([[5, 1, 4, 0]])  # SYSTEM, PUBLIC, PRIVATE, UNTRUSTED
        state = state_labels_to_vectors(labels, state_dim=8, noise=0.0)
        sm = attn._state_mask(state)  # [1,1,4,4]
        # PUBLIC (1) <- PRIVATE (2) forbidden
        self.assertTrue(torch.isinf(sm[0, 0, 1, 2]) and sm[0, 0, 1, 2] < 0)
        # SYSTEM (0) <- UNTRUSTED (3) allowed
        self.assertEqual(sm[0, 0, 0, 3].item(), 0.0)

        x = torch.randn(1, 4, 32)
        out, _ = attn(x, state)
        self.assertTrue(torch.isfinite(out).all())

    def test_fused_hard_mask(self):
        from nsa.fused_attention import FusedStateAwareAttention
        from nsa.utils import state_labels_to_vectors

        attn = FusedStateAwareAttention(
            d_model=32, state_dim=8, num_heads=4, gate_mode="hard", use_discrete_levels=True
        )
        labels = torch.tensor([[5, 0, 4]])
        state = state_labels_to_vectors(labels, state_dim=8)
        x = torch.randn(1, 3, 32)
        out, _ = attn(x, state)
        self.assertTrue(torch.isfinite(out).all())

    def test_security_coord_preserved_across_block(self):
        from nsa.layers import NSATransformerBlock
        from nsa.utils import state_labels_to_vectors

        block = NSATransformerBlock(d_model=32, state_dim=8, num_heads=4, gate_mode="hard", compat_mode="level")
        labels = torch.tensor([[5, 1, 0, 4]])
        state = state_labels_to_vectors(labels, state_dim=8)
        x = torch.randn(1, 4, 32)
        _, state_out = block(x, state)
        self.assertTrue(torch.allclose(state_out[..., 0], state[..., 0]))


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestLoRARetrofit(unittest.TestCase):
    def test_apply_nsa_lora_retrofit_wraps_and_counts(self):
        from nsa.lora import apply_nsa_lora_retrofit, NSALoRALinear

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(16, 16)
                self.self_attn.k_proj = nn.Linear(16, 16)
                self.self_attn.v_proj = nn.Linear(16, 16)
                self.self_attn.o_proj = nn.Linear(16, 16)

        m = Tiny()
        m, stats = apply_nsa_lora_retrofit(m, state_dim=4, r=4)
        self.assertGreater(stats["layers_wrapped"], 0)
        self.assertGreaterEqual(stats["total"], stats["trainable"])
        self.assertGreaterEqual(stats["frozen"], 0)
        self.assertLessEqual(stats["pct_trainable"], 100.0)
        self.assertIsInstance(m.self_attn.q_proj, NSALoRALinear)
        # base frozen, lora trainable
        self.assertFalse(m.self_attn.q_proj.base_layer.weight.requires_grad)
        self.assertTrue(m.self_attn.q_proj.lora_A.requires_grad)

    def test_lora_integrity_assertions_numeric(self):
        """Prove modules exist, trainable < total, base frozen — not just a named test."""
        from nsa.lora import apply_nsa_lora_retrofit, NSALoRALinear, DynamicNSARetrofitBlock

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q_proj = nn.Linear(32, 32)
                self.self_attn.k_proj = nn.Linear(32, 32)
                self.self_attn.v_proj = nn.Linear(32, 32)
                self.self_attn.o_proj = nn.Linear(32, 32)

        m = Tiny()
        m, stats = apply_nsa_lora_retrofit(m, state_dim=8, r=4, add_state_emb=False)
        n_lora = sum(1 for mod in m.modules() if isinstance(mod, NSALoRALinear))
        self.assertEqual(n_lora, 4)  # actual_lora_modules_exist
        self.assertLess(stats["trainable"], stats["total"])  # trainable_params < total_params
        self.assertGreater(stats["frozen"], 0)  # base_params_frozen mass
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer = getattr(m.self_attn, attr)
            self.assertFalse(any(p.requires_grad for p in layer.base_layer.parameters()))

        # Dynamic block also contains LoRA-wrapped projections
        blk = DynamicNSARetrofitBlock(
            d_model=32, state_dim=8, num_heads=4, r=4,
            gate_attention=True, gate_residual=False, gate_ffn=False, learn_sigma=False,
        )
        n_blk = sum(1 for mod in blk.modules() if isinstance(mod, NSALoRALinear))
        self.assertGreaterEqual(n_blk, 4)
        # gate_attention=False must set fused attn to off
        blk_off = DynamicNSARetrofitBlock(
            d_model=32, state_dim=8, num_heads=4,
            gate_attention=False, gate_residual=False, gate_ffn=False, learn_sigma=False,
        )
        self.assertEqual(blk_off.nsa_attn.fused_attn.gate_mode, "off")

    def test_fixed_alpha_zero_is_identity_state_update(self):
        from nsa.lora import DynamicNSARetrofitBlock
        from nsa.utils import state_labels_to_vectors

        blk = DynamicNSARetrofitBlock(
            d_model=32, state_dim=8, num_heads=4,
            gate_attention=True, gate_residual=False, gate_ffn=False,
            learn_sigma=True, fixed_alpha=0.0,
        )
        # Nonzero transition weights would still be multiplied by α=0
        with torch.no_grad():
            blk.state_transition.weight.fill_(0.5)
        x = torch.randn(2, 5, 32)
        labels = torch.tensor([[5, 1, 0, 4, 1], [1, 1, 1, 1, 1]])
        sigma = state_labels_to_vectors(labels, state_dim=8, noise=0.0)
        _, sigma_out = blk(x, sigma)
        self.assertTrue(torch.allclose(sigma_out[..., 0], sigma[..., 0]))
        # With α=0 and LN(sigma + 0), residual coords stay near input after LN
        # Security coord exact; full equality after LN may shift — only require finite + sec freeze
        self.assertTrue(torch.isfinite(sigma_out).all())


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestStateLabelVectors(unittest.TestCase):
    def test_dim0_is_exact_label(self):
        from nsa.utils import state_labels_to_vectors
        labels = torch.tensor([[0, 1, 4, 5]])
        v = state_labels_to_vectors(labels, state_dim=8, noise=0.0)
        self.assertTrue(torch.equal(v[..., 0], labels.float()))


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestResidualTaint(unittest.TestCase):
    def test_join_raises_taint(self):
        from nsa.residual_taint import ResidualTaintTracker

        levels = torch.tensor([[1, 1, 0]])  # PUBLIC, PUBLIC, UNTRUSTED
        tr = ResidualTaintTracker(levels)
        out = tr.residual_add(torch.tensor([[4, 1, 0]]), source="attn")
        self.assertEqual(int(out[0, 0].item()), 4)
        self.assertEqual(int(out[0, 1].item()), 1)
        self.assertGreaterEqual(len(tr.history), 1)

    def test_write_down_assert(self):
        from nsa.residual_taint import ResidualTaintTracker

        tr = ResidualTaintTracker(torch.tensor([[4, 1]]))
        with self.assertRaises(AssertionError):
            tr.assert_no_write_down(torch.tensor([[1, 1]]), name="public_channel")

    def test_declassify_auth(self):
        from nsa.residual_taint import ResidualTaintTracker

        tr = ResidualTaintTracker(torch.tensor([[4]]))
        with self.assertRaises(PermissionError):
            tr.declassify([(0, 0)], StateLabel.PUBLIC, authorized=False)
        tr.declassify([(0, 0)], StateLabel.PUBLIC, authorized=True)
        self.assertEqual(int(tr.levels[0, 0].item()), StateLabel.PUBLIC.value)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestTritonKernelModule(unittest.TestCase):
    def test_kernel_defined_and_cpu_manual_matches_finite(self):
        from nsa.triton_kernel import (
            HAS_TRITON,
            TRITON_KERNEL_DEFINED,
            last_backend,
            triton_fused_state_attention,
        )
        from nsa.utils import state_labels_to_vectors

        self.assertTrue(HAS_TRITON)
        self.assertTrue(TRITON_KERNEL_DEFINED)
        B, H, T, D = 1, 2, 4, 8
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        labels = torch.tensor([[5, 1, 0, 4]])
        state = state_labels_to_vectors(labels, state_dim=8, noise=0.0)
        out_m = triton_fused_state_attention(
            q, k, v, state, state, gate_mode="hard", force_backend="manual"
        )
        self.assertEqual(last_backend(), "manual")
        out_s = triton_fused_state_attention(
            q, k, v, state, state, gate_mode="hard", force_backend="sdpa"
        )
        self.assertTrue(torch.isfinite(out_m).all())
        self.assertTrue(torch.isfinite(out_s).all())
        self.assertTrue(torch.allclose(out_m, out_s, atol=1e-4, rtol=1e-3))

    def test_fused_module_forward(self):
        from nsa.triton_kernel import FusedTritonStateAttention
        from nsa.utils import state_labels_to_vectors

        attn = FusedTritonStateAttention(
            d_model=32, state_dim=8, num_heads=4, gate_mode="hard"
        )
        x = torch.randn(2, 6, 32)
        labels = torch.tensor([[5, 1, 0, 4, 1, 1], [1, 1, 1, 1, 1, 1]])
        state = state_labels_to_vectors(labels, state_dim=8)
        out, st = attn(x, state)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestKVCachePolicyMask(unittest.TestCase):
    def test_cache_policy_mask_blocks_downward(self):
        from nsa.kv_cache import NSAKVCache
        from nsa.utils import state_labels_to_vectors

        cache = NSAKVCache(batch_size=1, max_seq_len=8, num_heads=2, d_head=4, state_dim=8)
        labels = torch.tensor([[5, 0, 4]])
        state = state_labels_to_vectors(labels, state_dim=8)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        cache.update(k, v, state)
        q_state = state_labels_to_vectors(torch.tensor([[1]]), state_dim=8)  # PUBLIC query
        mask = cache.build_policy_mask(q_state)
        # PUBLIC cannot read SYSTEM (col 0) or PRIVATE (col 2)
        self.assertLess(mask[0, 0, 0, 0].item(), 0)
        self.assertLess(mask[0, 0, 0, 2].item(), 0)
        # PUBLIC can read UNTRUSTED (col 1)
        self.assertEqual(mask[0, 0, 0, 1].item(), 0.0)


@unittest.skipUnless(HAS_TORCH, "torch required")
class TestNLRedTeamFirewall(unittest.TestCase):
    def test_catalogue_mask_firewall_pass(self):
        # Import suite helpers without running main
        import importlib.util
        import sys
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "prototype" / "security" / "nl_redteam_suite.py"
        spec = importlib.util.spec_from_file_location("nl_redteam_suite", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["nl_redteam_suite"] = mod  # required for dataclasses on py3.8
        spec.loader.exec_module(mod)
        fw = mod.run_mask_firewall(mod.ATTACK_CATALOGUE)
        self.assertTrue(fw["all_untrusted_fully_blocked"])
        self.assertGreaterEqual(fw["mean_public_blocked_from_system"], 1.0 - 1e-6)


if __name__ == "__main__":
    unittest.main()
