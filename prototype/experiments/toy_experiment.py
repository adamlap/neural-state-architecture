"""
toy_experiment.py
=================
End-to-end proof-of-concept for Neural State Architecture.

Task:  Sequence classification — predict whether a sequence contains
       any PRIVATE token (binary classification).

Conservation law:  PRIVATE → PUBLIC is forbidden.
                   The state stream must not "declassify" tokens.

We compare two models:
    baseline — standard transformer (no state)
    nsa      — NSA transformer with state constraint loss

Metrics:
    accuracy        — task accuracy on both models
    violation_rate  — how often the NSA model's state stream violates
                      the conservation law (should decrease toward 0 during training)

Usage:
    python prototype/experiments/toy_experiment.py

Expected output:
    Both models reach similar accuracy; NSA model's violation_rate
    drops significantly below the baseline (which has no such concept).
"""

from __future__ import annotations

import sys
import os

# Allow importing from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nsa.algebra import StateLattice, StateLabel, DEFAULT_LATTICE
from nsa.layers import NSATransformer
from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
from nsa.utils import (
    count_parameters,
    make_privacy_dataset,
    state_labels_to_vectors,
    print_model_summary,
    print_lattice,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = dict(
    seed         = 42,
    n_samples    = 2000,
    seq_len      = 32,
    vocab_size   = 64,
    private_frac = 0.3,
    d_model      = 64,
    state_dim    = 8,
    num_layers   = 2,
    num_heads    = 4,
    batch_size   = 64,
    epochs       = 15,
    lr           = 3e-4,
    lambda_state = 0.5,       # weight of state constraint loss
    compat_mode  = "dot",     # 'dot', 'mlp', or 'level'
    gate_mode    = "soft",    # 'soft' or 'hard'
    device       = "cpu",     # change to 'cuda' if available
)


# ---------------------------------------------------------------------------
# Baseline: standard transformer (no state)
# ---------------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    """Standard transformer without state tracking — the control group."""

    def __init__(self, vocab_size, d_model, num_layers, num_heads, max_seq_len=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 2)

    def forward(self, tokens):
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        x    = self.tok_emb(tokens) + self.pos_emb(pos)
        x    = self.encoder(x)
        return self.head(x[:, 0])   # CLS-style: first token


# ---------------------------------------------------------------------------
# NSA model with classification head
# ---------------------------------------------------------------------------

class NSAClassifier(nn.Module):
    """NSA transformer + classification head."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.transformer = NSATransformer(
            vocab_size  = cfg["vocab_size"],
            d_model     = cfg["d_model"],
            state_dim   = cfg["state_dim"],
            num_layers  = cfg["num_layers"],
            num_heads   = cfg["num_heads"],
            max_seq_len = cfg["seq_len"] + 1,
            compat_mode = cfg["compat_mode"],
            gate_mode   = cfg["gate_mode"],
        )
        self.head = nn.Linear(cfg["d_model"], 2)

    def forward(
        self, tokens: torch.Tensor, state_init: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, states_in, states_out)."""
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device).unsqueeze(0)
        # states_in: the initial state (before the transformer)
        states_in = state_init.clone()
        x, states_out = self.transformer(tokens, state_init=state_init)
        logits = self.head(x[:, 0])   # CLS token
        return logits, states_in, states_out


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_baseline(cfg: dict, loader: DataLoader) -> List[float]:
    model = BaselineTransformer(
        vocab_size  = cfg["vocab_size"],
        d_model     = cfg["d_model"],
        num_layers  = cfg["num_layers"],
        num_heads   = cfg["num_heads"],
    ).to(cfg["device"])

    opt      = optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn  = nn.CrossEntropyLoss()
    accs     = []

    print(f"\n{'-'*50}")
    print("Training BASELINE transformer")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(cfg["epochs"]):
        model.train()
        correct = total = 0
        for tokens, _, targets in loader:
            tokens  = tokens.to(cfg["device"])
            targets = targets.to(cfg["device"])
            logits  = model(tokens)
            loss    = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            correct += (logits.argmax(-1) == targets).sum().item()
            total   += targets.size(0)
        acc = correct / total
        accs.append(acc)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  acc={acc:.3f}")

    return accs


def train_nsa(cfg: dict, loader: DataLoader) -> Tuple[List[float], List[float]]:
    model   = NSAClassifier(cfg).to(cfg["device"])
    lattice = DEFAULT_LATTICE

    sem_loss   = SemanticLoss("cross_entropy")
    state_loss = StateConstraintLoss(
        state_dim = cfg["state_dim"],
        lattice   = lattice,
        mode      = "level",
        margin    = 0.05,
    )
    criterion  = NSALoss(
        semantic_loss = sem_loss,
        state_loss    = state_loss,
        lambda_init   = cfg["lambda_state"],
    )

    opt  = optim.AdamW(model.parameters(), lr=cfg["lr"])
    accs = []
    viols = []

    print(f"\n{'-'*50}")
    print("Training NSA transformer")
    print_model_summary(model)

    for epoch in range(cfg["epochs"]):
        model.train()
        correct = total = 0
        viol_sum = viol_n = 0

        for tokens, state_labels, targets in loader:
            tokens       = tokens.to(cfg["device"])
            state_labels = state_labels.to(cfg["device"])
            targets      = targets.to(cfg["device"])

            # Convert integer labels → continuous state vectors
            state_init = state_labels_to_vectors(
                state_labels, state_dim=cfg["state_dim"]
            )

            logits, states_in, states_out = model(tokens, state_init)
            loss, metrics = criterion(logits, targets, states_in, states_out)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Track accuracy
            correct += (logits.argmax(-1) == targets).sum().item()
            total   += targets.size(0)

            # Track violation rate
            vr = state_loss.violation_rate(states_in, states_out)
            viol_sum += vr
            viol_n   += 1

        acc  = correct / total
        viol = viol_sum / viol_n
        accs.append(acc)
        viols.append(viol)

        # Adaptive λ update
        criterion.update_lambda(viol)

        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']}  "
                f"acc={acc:.3f}  "
                f"viol={viol:.4f}  "
                f"lambda={criterion.lam.item():.3f}"
            )

    return accs, viols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(CFG["seed"])

    print("=" * 50)
    print("Neural State Architecture — Toy Experiment")
    print("Task: Binary classification (contains PRIVATE token?)")
    print("Conservation law: PRIVATE -> PUBLIC is forbidden")
    print("=" * 50)

    print("\nLattice transition rules:")
    print_lattice()

    # Generate data
    tokens, state_labels, targets = make_privacy_dataset(
        n_samples    = CFG["n_samples"],
        seq_len      = CFG["seq_len"],
        vocab_size   = CFG["vocab_size"],
        private_frac = CFG["private_frac"],
        device       = CFG["device"],
    )

    dataset = TensorDataset(tokens, state_labels, targets)

    # Split train/val (80/20)
    n_train  = int(0.8 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)

    print(f"\nDataset: {n_train} train / {n_val} val samples")
    print(f"Private token fraction: {CFG['private_frac']:.0%}")

    t0 = time.time()

    baseline_accs = train_baseline(CFG, train_loader)
    nsa_accs, nsa_viols = train_nsa(CFG, train_loader)

    elapsed = time.time() - t0

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Training time: {elapsed:.1f}s")
    print()
    print(f"{'Metric':<35} {'Baseline':>10} {'NSA':>10}")
    print(f"{'-' * 55}")
    print(f"{'Final accuracy':<35} {baseline_accs[-1]:>10.3f} {nsa_accs[-1]:>10.3f}")
    print(f"{'Peak accuracy':<35} {max(baseline_accs):>10.3f} {max(nsa_accs):>10.3f}")
    print(f"{'Final violation rate':<35} {'N/A':>10} {nsa_viols[-1]:>10.4f}")
    print(f"{'Min violation rate':<35} {'N/A':>10} {min(nsa_viols):>10.4f}")
    print()

    if nsa_viols[-1] < 0.05:
        print("[OK] Conservation law is well-respected (violation rate < 5%)")
    else:
        print(f"[!] Conservation law violation rate is {nsa_viols[-1]:.1%} - consider increasing lambda")

    delta_acc = nsa_accs[-1] - baseline_accs[-1]
    if abs(delta_acc) < 0.05:
        print(f"[OK] Task accuracy is comparable (delta={delta_acc:+.3f})")
    elif delta_acc > 0:
        print(f"[OK] NSA outperforms baseline by {delta_acc:.3f}")
    else:
        print(f"[!] NSA underperforms baseline by {-delta_acc:.3f} - state constraint may be too strong")

    print()
    print("Interpretation:")
    print("  The NSA model maintains task performance while explicitly")
    print("  respecting information-flow constraints in its state stream.")
    print("  This is the key property absent from standard architectures.")


if __name__ == "__main__":
    main()
