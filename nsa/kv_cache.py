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

    def predict_next_state(self, transition_fn) -> torch.Tensor:
        """Autoregressively computes state for next token: σ_{t+1} = transition_fn(σ_t)."""
        if self.seq_len == 0:
            return torch.zeros((self.batch_size, 1, self.state_dim), device=self.device, dtype=self.dtype)
        last_state = self.state_cache[:, self.seq_len - 1:self.seq_len, :]
        return transition_fn(last_state)

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.state_cache.zero_()
        self.seq_len = 0

    def get_state_labels(self) -> torch.Tensor:
        """Discrete security labels for cached tokens: round(σ[..., 0])."""
        if self.seq_len == 0:
            return torch.zeros((self.batch_size, 0), device=self.device, dtype=torch.long)
        return self.state_cache[:, : self.seq_len, 0].round().long().clamp(0, 5)

    def build_policy_mask(
        self,
        query_state: torch.Tensor,
        *,
        forbidden_value: float = float("-inf"),
        lattice=None,
    ) -> torch.Tensor:
        """Additive NSA mask for current queries against the full K cache.

        Parameters
        ----------
        query_state : Tensor [B, T_q, state_dim]
        """
        from nsa.algebra import DEFAULT_LATTICE, build_label_attention_mask

        lattice = lattice or DEFAULT_LATTICE
        q_lab = query_state[..., 0].round().long().clamp(0, 5)
        k_lab = self.get_state_labels()
        if k_lab.shape[-1] == 0:
            B, Tq = q_lab.shape[0], q_lab.shape[1]
            return torch.zeros(B, 1, Tq, 0, device=self.device, dtype=self.dtype)
        return build_label_attention_mask(
            q_lab, k_lab, lattice=lattice, forbidden_value=forbidden_value
        ).to(device=self.device, dtype=self.dtype)

    def clone_view(self) -> "NSAKVCache":
        """Shallow metadata clone sharing storage (for multi-sequence fans)."""
        other = NSAKVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_heads=self.num_heads,
            d_head=self.d_head,
            state_dim=self.state_dim,
            device=self.device,
            dtype=self.dtype,
        )
        other.k_cache = self.k_cache
        other.v_cache = self.v_cache
        other.state_cache = self.state_cache
        other.seq_len = self.seq_len
        return other
