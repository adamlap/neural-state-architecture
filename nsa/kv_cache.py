"""
nsa.kv_cache
============
KV-Cache State Tracking for High-Throughput Inference (vLLM / sglang).

Maintains Key, Value, and State-Vector caches across autoregressive generation steps.
Ensures policy governance state vectors σ_1:t are tracked per sequence layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch


class NSAKVCache:
    """Layer-wise Key-Value-State Cache for high-performance NSA inference engines."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        d_head: int,
        state_dim: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_head = d_head
        self.state_dim = state_dim
        self.device = device
        self.dtype = dtype

        # Allocated Cache Tensors
        self.k_cache = torch.zeros((batch_size, num_heads, max_seq_len, d_head), device=device, dtype=dtype)
        self.v_cache = torch.zeros((batch_size, num_heads, max_seq_len, d_head), device=device, dtype=dtype)
        self.state_cache = torch.zeros((batch_size, max_seq_len, state_dim), device=device, dtype=dtype)

        self.seq_len = 0

    def update(
        self,
        k_new: torch.Tensor,     # [B, H, T_new, d_head]
        v_new: torch.Tensor,     # [B, H, T_new, d_head]
        state_new: torch.Tensor, # [B, T_new, state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Appends new tokens to key, value, and state caches.

        Returns slice up to current length: (K, V, State)
        """
        B, H, T_new, _ = k_new.shape
        start_idx = self.seq_len
        end_idx = start_idx + T_new

        self.k_cache[:, :, start_idx:end_idx, :] = k_new
        self.v_cache[:, :, start_idx:end_idx, :] = v_new
        self.state_cache[:, start_idx:end_idx, :] = state_new

        self.seq_len = end_idx

        return (
            self.k_cache[:, :, :end_idx, :],
            self.v_cache[:, :, :end_idx, :],
            self.state_cache[:, :end_idx, :],
        )

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.state_cache.zero_()
        self.seq_len = 0
