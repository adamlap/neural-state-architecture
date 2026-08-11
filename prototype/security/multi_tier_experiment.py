"""
prototype/security/multi_tier_experiment.py
==================================
Multi-Tier Security Lattice Benchmark for Neural State Architecture.

Task:
    Multi-level security governance across 4 lattice security levels:
        UNTRUSTED (0) < PUBLIC (1) < CONFIDENTIAL (3) < PRIVATE (4)

    Evaluates state propagation fidelity, multi-tier conservation violation rates,
    and state transition precision across deep transformer layers.

Usage:
    python prototype/security/multi_tier_experiment.py
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Allow parent module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nsa.algebra import StateLabel, DEFAULT_LATTICE
from nsa.layers import NSATransformer
from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
from nsa.utils import count_parameters, print_model_summary, print_lattice


def make_multitier_dataset(
    n_samples: int = 2000,
    seq_len: int = 32,
    vocab_size: int = 64,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a multi-tier dataset with UNTRUSTED, PUBLIC, CONFIDENTIAL, and PRIVATE labels.

    State labels are assigned per-token:
      - 0: UNTRUSTED
      - 1: PUBLIC
      - 3: CONFIDENTIAL
      - 4: PRIVATE
    Target is multi-class sequence level (highest security tier present in sequence).
    """
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    
    # Assign labels based on token values
    state_labels = torch.full((n_samples, seq_len), StateLabel.PUBLIC.value, dtype=torch.long, device=device)
    
    # Set UNTRUSTED (0) for token values < 10
    state_labels[tokens < 10] = StateLabel.UNTRUSTED.value
    # Set CONFIDENTIAL (3) for token values > 45
    state_labels[tokens > 45] = StateLabel.CONFIDENTIAL.value
    # Set PRIVATE (4) for token values > 55
    state_labels[tokens > 55] = StateLabel.PRIVATE.value

    # Target: highest security label in the sequence (0, 1, 3, or 4 mapped to 0..3)
    max_labels, _ = state_labels.max(dim=-1)
    
    # Map raw label values (0, 1, 3, 4) to class indices (0, 1, 2, 3)
    label_map = {
        StateLabel.UNTRUSTED.value: 0,
        StateLabel.PUBLIC.value: 1,
        StateLabel.CONFIDENTIAL.value: 2,
        StateLabel.PRIVATE.value: 3,
    }
    targets = torch.zeros(n_samples, dtype=torch.long, device=device)
    for i in range(n_samples):
        targets[i] = label_map.get(max_labels[i].item(), 1)

    return tokens, state_labels, targets


class MultiTierNSAClassifier(nn.Module):
    """NSA Model for Multi-Tier Security Lattice Classification."""

    def __init__(self, vocab_size: int, d_model: int, state_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.nsa = NSATransformer(
            d_model=d_model, state_dim=state_dim, num_layers=num_layers, num_heads=num_heads,
            compat_mode="level", gate_mode="soft", lattice=DEFAULT_LATTICE
        )
        self.head = nn.Linear(d_model, 4)  # 4 security classes

    def forward(self, tokens: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_out, state_out = self.nsa(tokens, state_init=states)
        logits = self.head(x_out.mean(dim=1))
        return logits, x_out, state_out


def run_multitier_experiment(
    n_samples: int = 2000,
    seq_len: int = 32,
    d_model: int = 64,
    state_dim: int = 8,
    epochs: int = 15,
    device: str = "cpu"
):
    print("=" * 70)
    print("  NSA Benchmark: Multi-Tier Security Lattice Governance")
    print("  Tiers: UNTRUSTED (0) < PUBLIC (1) < CONFIDENTIAL (3) < PRIVATE (4)")
    print("=" * 70)

    # 1. Dataset
    tokens, state_labels, targets = make_multitier_dataset(
        n_samples=n_samples, seq_len=seq_len, device=device
    )

    # Convert discrete state labels to state vectors
    # Simple one-hot style vector for state stream
    B, T = state_labels.shape
    state_vectors = torch.zeros(B, T, state_dim, device=device)
    for b in range(B):
        for t in range(T):
            lbl_val = state_labels[b, t].item()
            state_vectors[b, t, min(lbl_val, state_dim - 1)] = 1.0

    split = int(n_samples * 0.8)
    train_dataset = TensorDataset(tokens[:split], state_vectors[:split], state_labels[:split], targets[:split])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 2. Model & Loss
    model = MultiTierNSAClassifier(vocab_size=64, d_model=d_model, state_dim=state_dim, num_layers=2, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    state_loss_fn = StateConstraintLoss(state_dim=state_dim, lattice=DEFAULT_LATTICE)
    nsa_loss_fn = NSALoss(
        semantic_loss=SemanticLoss(nn.CrossEntropyLoss()),
        state_loss=state_loss_fn,
        lambda_init=0.5
    )

    print("\nTraining Multi-Tier NSA Model...")
    print_model_summary(model)

    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        for b_tok, b_svec, b_slab, b_tgt in train_loader:
            optimizer.zero_grad()
            logits, x_out, state_out = model(b_tok, b_svec)
            total_loss, metrics = nsa_loss_fn(logits, b_tgt, b_svec, state_out)
            total_loss.backward()
            optimizer.step()
            total_loss_epoch += total_loss.item()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_logits, val_x_out, val_s_out = model(tokens[split:], state_vectors[split:])
                acc = (val_logits.argmax(dim=-1) == targets[split:]).float().mean().item()
                viol_rate = state_loss_fn.violation_rate(state_vectors[split:], val_s_out)
            print(f"  Epoch {ep:>2d}/{epochs} | Loss: {total_loss_epoch / len(train_loader):.4f} | Val Acc: {acc * 100:.2f}% | Violation Rate: {viol_rate * 100:.2f}%")

    duration = time.time() - start_time
    print("\n" + "=" * 70)
    print("  MULTI-TIER BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Training Time                     : {duration:.2f}s")
    print(f"  Final Multi-Class Task Accuracy   : {acc * 100:.2f}%")
    print(f"  Final State Violation Rate        : {viol_rate * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSA Multi-Tier Security Lattice Benchmark")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--d-model", type=int, default=64, help="Model hidden dimension")
    args = parser.parse_args()

    run_multitier_experiment(
        n_samples=args.n_samples,
        epochs=args.epochs,
        d_model=args.d_model
    )
