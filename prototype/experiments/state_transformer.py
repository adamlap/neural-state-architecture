"""
prototype/experiments/state_transformer.py
===============================
Reference prototype script demonstrating how to instantiate and use
NSATransformerBlock and StateAwareAttention from the `nsa` core package.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from nsa import DEFAULT_LATTICE, NSATransformerBlock, print_model_summary


def main():
    print("=" * 60)
    print("NSA Prototype Demonstration")
    print("=" * 60)

    batch_size = 2
    seq_len = 16
    d_model = 128
    state_dim = 8

    # Inputs: semantic activations x, and state vectors state
    x = torch.randn(batch_size, seq_len, d_model)
    state = torch.randn(batch_size, seq_len, state_dim)

    print(f"Input semantic tensor shape: {x.shape}")
    print(f"Input state tensor shape:    {state.shape}")

    # Instantiate NSA Transformer Block
    block = NSATransformerBlock(
        d_model=d_model,
        state_dim=state_dim,
        num_heads=8,
        compat_mode="dot",
        gate_mode="soft",
        lattice=DEFAULT_LATTICE
    )

    y, next_state = block(x, state)

    print("\nForward pass completed successfully!")
    print(f"Output semantic tensor shape: {y.shape}")
    print(f"Output state tensor shape:    {next_state.shape}")

    print("\nModel Parameter Breakdown:")
    print_model_summary(block)

if __name__ == "__main__":
    main()
