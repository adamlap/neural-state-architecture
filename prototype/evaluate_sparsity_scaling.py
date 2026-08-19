"""
prototype/evaluate_sparsity_scaling.py
======================================
NSA 2.1 Structured Sparsity & State-Gated Tile-Skipping Evaluation.

Evaluates performance scaling across partial policy constraints:
Forbidden Edge Ratios: [0% (unconstrained), 10%, 25%, 50%, 75%, 90%, 95%]

Key Architectural Discovery:
    State constraints define structured block sparsity.
    When a K-tile is fully forbidden (max(sigma_k) > sigma_q), QK matmul, softmax,
    and PV tile memory load can be SKIPPED entirely.

Measures:
- Latency (ms)
- Generation Throughput (tokens/s)
- Actual Attention Edge Sparsity (%)
- Tile Skip Efficiency (%)
- Speedup Relative to Dense Baseline
"""

from __future__ import annotations

import time
import math
import torch
import torch.nn.functional as F

from nsa.algebra import StateLabel
from nsa.triton_kernel import triton_fused_state_attention


def evaluate_sparsity_scaling():
    print("=" * 115)
    print("  NSA 2.1 STRUCTURED SPARSITY & STATE-GATED TILE-SKIPPING EVALUATION")
    print("=" * 115)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    batch_size = 1
    num_heads = 8
    seq_len = 4096
    head_dim = 64
    tile_size = 32
    iters = 10

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Constraint ratios to evaluate
    forbidden_ratios = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    print(f"\n{'Target Forbidden':<18} | {'Measured Sparsity':<18} | {'Skipped Tiles':<16} | {'Latency (ms)':<14} | {'Throughput (tok/s)':<18} | {'Relative Speed'}")
    print("-" * 115)

    for ratio in forbidden_ratios:
        # Construct state vector where ratio% of key tokens are higher clearance (e.g. SYSTEM vs PUBLIC)
        # Query tokens: all PUBLIC (can only read PUBLIC, forbidden from reading SYSTEM)
        q_labels = torch.full((batch_size, seq_len), StateLabel.PUBLIC.value, device=device)
        k_labels = q_labels.clone()
        
        # Mark ratio% of keys as SYSTEM
        num_forbidden = int(seq_len * ratio)
        if num_forbidden > 0:
            k_labels[:, :num_forbidden] = StateLabel.SYSTEM.value

        # Calculate actual forbidden edge count (taking causal into account)
        offs_q = torch.arange(seq_len, device=device).unsqueeze(-1)
        offs_k = torch.arange(seq_len, device=device).unsqueeze(-2)
        compat = (q_labels.unsqueeze(-1) >= k_labels.unsqueeze(-2)) & (offs_q >= offs_k)
        total_causal_edges = (seq_len * (seq_len + 1)) // 2
        active_edges = int(compat.sum().item())
        actual_sparsity = ((total_causal_edges - active_edges) / total_causal_edges) * 100.0

        # Calculate tile skip percentage (blocks of 32 keys where all keys are SYSTEM)
        num_tiles = seq_len // tile_size
        skipped_tiles = 0
        for t_idx in range(num_tiles):
            tile_slice = k_labels[0, t_idx * tile_size : (t_idx + 1) * tile_size]
            if (tile_slice == StateLabel.SYSTEM.value).all():
                skipped_tiles += 1
        tile_skip_pct = (skipped_tiles / num_tiles) * 100.0

        # Measure simulated state-aware sparse attention forward
        # (Sparse execution skips computation on fully masked tiles)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            # Compute on non-skipped tiles
            out = triton_fused_state_attention(q, k, v, q_states=q_labels, k_states=k_labels, is_causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat = ((time.perf_counter() - t0) / iters) * 1000.0
        tok_s = (batch_size * seq_len) / (lat / 1000.0)

        # Baseline unconstrained reference at 0%
        if ratio == 0.0:
            base_lat = lat

        speedup = base_lat / lat if lat > 0 else 1.0

        print(f"{ratio * 100.0:>6.1f}% forbidden   | {actual_sparsity:>15.1f}%   | {tile_skip_pct:>13.1f}%   | {lat:>10.2f} ms | {tok_s:>14.1f} tok/s | {speedup:>12.2f}x")

    print("=" * 115)
    print("  Key Takeaway: Policy constraints naturally define structured block sparsity.")
    print("  High-restriction workloads (e.g. 75%-90% private context) allow aggressive tile skipping.")
    print("=" * 115)


if __name__ == "__main__":
    evaluate_sparsity_scaling()
