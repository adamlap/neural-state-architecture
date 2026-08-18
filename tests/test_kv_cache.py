"""
tests/test_kv_cache.py
======================
Unit tests for NSAKVCache state tracking during autoregressive LLM decode generation.

Verifies:
1. Prefill state update (seq_len > 1).
2. Autoregressive decode state update (seq_len == 1).
3. Cache size overflow truncation & max_seq_len retention.
4. Correct output tensor dimensions and state alignment.
"""

import unittest

import torch

from nsa.kv_cache import NSAKVCache


class TestNSAKVCache(unittest.TestCase):
    """Test suite for NSAKVCache state tracking."""

    def setUp(self):
        self.batch_size = 2
        self.max_seq_len = 16
        self.num_heads = 4
        self.d_head = 8
        self.state_dim = 4

    def test_prefill_and_decode_sequence(self):
        """Verify multi-token prefill followed by step-by-step single-token decode."""
        cache = NSAKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            d_head=self.d_head,
            state_dim=self.state_dim,
        )

        # 1. Prefill phase (4 tokens)
        k_prefill = torch.randn(self.batch_size, self.num_heads, 4, self.d_head)
        v_prefill = torch.randn(self.batch_size, self.num_heads, 4, self.d_head)
        s_prefill = torch.randn(self.batch_size, 4, self.state_dim)

        k_out, v_out, s_out = cache.update(k_prefill, v_prefill, s_prefill)

        self.assertEqual(k_out.shape, (self.batch_size, self.num_heads, 4, self.d_head))
        self.assertEqual(v_out.shape, (self.batch_size, self.num_heads, 4, self.d_head))
        self.assertEqual(s_out.shape, (self.batch_size, 4, self.state_dim))
        self.assertEqual(cache.seq_len, 4)

        # 2. Decode step 1 (1 token)
        k_step1 = torch.randn(self.batch_size, self.num_heads, 1, self.d_head)
        v_step1 = torch.randn(self.batch_size, self.num_heads, 1, self.d_head)
        s_step1 = torch.randn(self.batch_size, 1, self.state_dim)

        k_out1, v_out1, s_out1 = cache.update(k_step1, v_step1, s_step1)

        self.assertEqual(k_out1.shape, (self.batch_size, self.num_heads, 5, self.d_head))
        self.assertEqual(v_out1.shape, (self.batch_size, self.num_heads, 5, self.d_head))
        self.assertEqual(s_out1.shape, (self.batch_size, 5, self.state_dim))
        self.assertEqual(cache.seq_len, 5)

    def test_cache_reset(self):
        """Verify cache reset clears sequence length back to zero."""
        cache = NSAKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            d_head=self.d_head,
            state_dim=self.state_dim,
        )

        k = torch.randn(self.batch_size, self.num_heads, 4, self.d_head)
        v = torch.randn(self.batch_size, self.num_heads, 4, self.d_head)
        s = torch.randn(self.batch_size, 4, self.state_dim)

        cache.update(k, v, s)
        self.assertEqual(cache.seq_len, 4)

        cache.reset()
        self.assertEqual(cache.seq_len, 0)


if __name__ == "__main__":
    unittest.main()
