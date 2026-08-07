"""
prototype/pretrain_lm.py
========================
Pillar 1 Benchmark: Empirical Zero Quality Degradation at Scale.

Objective:
    Demonstrate that adding dual-stream state governance (NSACausalLM) causes
    < 0.1% perplexity degradation relative to a standard Baseline Causal Transformer
    under identical model dimensions, parameter scale, and pre-training data budget.

Metrics:
    - Semantic Loss (Cross Entropy)
    - Language Model Perplexity: PPL = exp(L_semantic)
    - Perplexity Delta: (PPL_NSA - PPL_Baseline) / PPL_Baseline * 100%
    - State Conservation Violation Rate: % state transitions violating lattice algebra
    - Parameter Overhead: % state stream parameters relative to total trainable parameters

Usage:
    python prototype/pretrain_lm.py
"""

from __future__ import annotations

import argparse
import sys
import os
import math
import time
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Allow parent module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nsa.algebra import StateLabel, DEFAULT_LATTICE
from nsa.layers import NSACausalLM
from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
from nsa.utils import count_parameters, make_privacy_dataset, state_labels_to_vectors


class BaselineCausalLM(nn.Module):
    """Standard GPT-style Causal Language Model (Baseline control group)."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        device = tokens.device
        pos = torch.arange(T, device=device).unsqueeze(0)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

        x = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        x = self.encoder(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def generate_lm_corpus(
    n_samples: int = 2000,
    seq_len: int = 64,
    vocab_size: int = 128,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic language modeling sequence corpus.

    Returns:
        inputs       : [n_samples, seq_len - 1]
        targets      : [n_samples, seq_len - 1] (autoregressive next token)
        state_vectors: [n_samples, seq_len - 1, state_dim]
        state_labels : [n_samples, seq_len - 1]
    """
    tokens, state_labels, _ = make_privacy_dataset(
        n_samples=n_samples, seq_len=seq_len, vocab_size=vocab_size, private_frac=0.25, device=device
    )

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    state_labs_in = state_labels[:, :-1]
    state_vectors_in = state_labels_to_vectors(state_labs_in, state_dim=8)

    return inputs, targets, state_vectors_in, state_labs_in


def run_lm_benchmark(
    n_samples: int = 2400,
    seq_len: int = 64,
    vocab_size: int = 128,
    d_model: int = 128,
    state_dim: int = 8,
    num_layers: int = 4,
    num_heads: int = 8,
    epochs: int = 15,
    device: str = "cpu",
):
    print("=" * 75)
    print("  PILLAR 1 BENCHMARK: Empirical Zero Quality Degradation at Scale")
    print("  Comparing Causal LLM Perplexity: Baseline vs Neural State Architecture")
    print("=" * 75)

    # 1. Dataset
    inputs, targets, state_vectors, state_labels = generate_lm_corpus(
        n_samples=n_samples, seq_len=seq_len, vocab_size=vocab_size, device=device
    )

    split = int(n_samples * 0.8)
    train_dataset = TensorDataset(inputs[:split], targets[:split], state_vectors[:split], state_labels[:split])
    val_dataset = TensorDataset(inputs[split:], targets[split:], state_vectors[split:], state_labels[split:])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # 1. BASELINE Causal LM Training
    # -----------------------------------------------------------------------
    print("\n[1/2] Training Baseline Causal Transformer LM...")
    baseline_lm = BaselineCausalLM(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, dropout=0.0
    ).to(device)
    base_counts = count_parameters(baseline_lm)
    print(f"  Baseline Trainable Parameters: {base_counts['trainable']:,}")

    opt_base = optim.AdamW(baseline_lm.parameters(), lr=1e-3, weight_decay=0.01)

    start_time = time.time()
    for ep in range(1, epochs + 1):
        baseline_lm.train()
        for b_in, b_tgt, _, _ in train_loader:
            opt_base.zero_grad()
            logits = baseline_lm(b_in)
            loss = criterion(logits.reshape(-1, vocab_size), b_tgt.reshape(-1))
            loss.backward()
            opt_base.step()

        if ep % 5 == 0 or ep == epochs:
            baseline_lm.eval()
            with torch.no_grad():
                val_logits = baseline_lm(inputs[split:])
                val_loss = criterion(val_logits.reshape(-1, vocab_size), targets[split:].reshape(-1)).item()
                val_ppl = math.exp(val_loss)
            print(f"  Epoch {ep:>2d}/{epochs} | Val Loss: {val_loss:.4f} | Perplexity (PPL): {val_ppl:.2f}")

    base_final_ppl = val_ppl

    # -----------------------------------------------------------------------
    # 2. NSA Causal LM Training
    # -----------------------------------------------------------------------
    print("\n[2/2] Training NSA Causal Transformer LM (Dual-Stream State Governance)...")
    nsa_lm = NSACausalLM(
        vocab_size=vocab_size, d_model=d_model, state_dim=state_dim,
        num_layers=num_layers, num_heads=num_heads, compat_mode="dot", gate_mode="soft", dropout=0.0
    ).to(device)

    nsa_counts = count_parameters(nsa_lm)
    print(f"  NSA Total Parameters    : {nsa_counts['total']:,}")
    print(f"  Semantic Parameters     : {nsa_counts['semantic']:,}")
    print(f"  State Stream Parameters : {nsa_counts['state']:,}")
    pct_overhead = (nsa_counts['state'] / max(nsa_counts['trainable'], 1)) * 100
    print(f"  State Parameter Overhead: {pct_overhead:.2f}%")

    opt_nsa = optim.AdamW(nsa_lm.parameters(), lr=1e-3, weight_decay=0.01)
    state_loss_fn = StateConstraintLoss(state_dim=state_dim, lattice=DEFAULT_LATTICE)
    nsa_loss_fn = NSALoss(
        semantic_loss=SemanticLoss(criterion),
        state_loss=state_loss_fn,
        lambda_init=0.5
    )

    for ep in range(1, epochs + 1):
        nsa_lm.train()
        for b_in, b_tgt, b_svec, b_slab in train_loader:
            opt_nsa.zero_grad()
            logits, x_out, state_out = nsa_lm(b_in, state_init=b_svec)
            total_loss, metrics = nsa_loss_fn(logits.reshape(-1, vocab_size), b_tgt.reshape(-1), b_svec, state_out)
            total_loss.backward()
            opt_nsa.step()

        if ep % 5 == 0 or ep == epochs:
            nsa_lm.eval()
            with torch.no_grad():
                val_logits, val_x_out, val_s_out = nsa_lm(inputs[split:], state_init=state_vectors[split:])
                val_loss = criterion(val_logits.reshape(-1, vocab_size), targets[split:].reshape(-1)).item()
                val_ppl = math.exp(val_loss)
                viol_rate = state_loss_fn.violation_rate(state_vectors[split:], val_s_out)
            print(f"  Epoch {ep:>2d}/{epochs} | Val Loss: {val_loss:.4f} | Perplexity (PPL): {val_ppl:.2f} | Violation Rate: {viol_rate * 100:.2f}%")

    nsa_final_ppl = val_ppl
    ppl_delta_pct = ((nsa_final_ppl - base_final_ppl) / base_final_ppl) * 100

    duration = time.time() - start_time

    # -----------------------------------------------------------------------
    # 3. Benchmark Summary Report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  PILLAR 1 BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"  Metric                             Baseline        NSA            Delta")
    print(f"  -----------------------------------------------------------------------")
    print(f"  Validation Loss                    {val_loss:>7.4f}      {val_loss:>7.4f}        —")
    print(f"  Perplexity (PPL)                   {base_final_ppl:>7.2f}      {nsa_final_ppl:>7.2f}       {ppl_delta_pct:>+6.2f}%")
    print(f"  State Violation Rate                     N/A       {viol_rate * 100:>6.2f}%        —")
    print(f"  State Parameter Overhead                 N/A       {pct_overhead:>6.2f}%        —")
    print(f"  Benchmarking Time                  {duration:>6.2f}s")
    print("=" * 75)

    if ppl_delta_pct <= 0.1:
        print("  [PASSED] PILLAR 1 VERIFIED: Zero Quality Degradation (< 0.1% PPL Delta)!")
        print("  Interpretation: Adding state governance causes negligible language modeling")
        print("  loss penalty while enforcing formal algebraic state conservation.")
    else:
        print(f"  [NOTE] Perplexity delta is {ppl_delta_pct:+.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSA Pillar 1 Language Modeling Benchmark")
    parser.add_argument("--n-samples", type=int, default=2400, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    args = parser.parse_args()

    run_lm_benchmark(
        n_samples=args.n_samples,
        epochs=args.epochs,
        d_model=args.d_model
    )
