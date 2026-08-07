"""
nsa.utils
=========
Helpers: parameter counting, state visualisation, lattice diagram printing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

from nsa.algebra import StateLabel, StateLattice, DEFAULT_LATTICE


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return parameter counts split by semantic vs. state streams."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    state_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if any(k in name for k in ("state", "transition", "V.weight", "level_proj", "state_gate"))
    )

    return {
        "total":     total,
        "trainable": trainable,
        "state":     state_params,
        "semantic":  trainable - state_params,
    }


def print_model_summary(model: nn.Module) -> None:
    counts = count_parameters(model)
    print("=" * 50)
    print("NSA Model Parameter Summary")
    print("=" * 50)
    print(f"  Total parameters  : {counts['total']:>10,}")
    print(f"  Trainable         : {counts['trainable']:>10,}")
    print(f"  Semantic stream   : {counts['semantic']:>10,}")
    print(f"  State stream      : {counts['state']:>10,}")
    pct = 100 * counts['state'] / max(counts['trainable'], 1)
    print(f"  State overhead    : {pct:>9.1f}%")
    print("=" * 50)


# ---------------------------------------------------------------------------
# State visualisation
# ---------------------------------------------------------------------------

def state_level_heatmap(
    states: torch.Tensor,        # [B, T, state_dim]
    level_proj: nn.Linear,       # the level projection from StateConstraintLoss
    tokens: Optional[List[str]] = None,
) -> None:
    """Print an ASCII heatmap of per-token state levels.

    Useful for visualising which tokens have high vs. low restriction levels
    during inference.
    """
    with torch.no_grad():
        levels = level_proj(states).squeeze(-1)   # [B, T]
        B, T = levels.shape
        for b in range(min(B, 3)):   # show up to 3 samples
            print(f"\nSample {b} state levels:")
            row = ""
            for t in range(T):
                v = levels[b, t].item()
                # Map level to a shade character
                shade = "░▒▓█"[min(3, max(0, int(v * 2)))]
                tok   = (tokens[t] if tokens and t < len(tokens) else str(t))
                row  += f"[{shade}{tok[:4]:4s}]"
            print(row)


def print_lattice(lattice: StateLattice = DEFAULT_LATTICE) -> None:
    """Pretty-print the transition table for a given lattice."""
    print(lattice.summary())


# ---------------------------------------------------------------------------
# Generating synthetic data (for toy experiments)
# ---------------------------------------------------------------------------

def make_privacy_dataset(
    n_samples:    int   = 1000,
    seq_len:      int   = 32,
    vocab_size:   int   = 64,
    private_frac: float = 0.3,
    device:       str   = "cpu",
):
    """Generate a synthetic token dataset with PUBLIC/PRIVATE labels.

    Returns
    -------
    tokens      : LongTensor [n_samples, seq_len]
    state_labels: LongTensor [n_samples, seq_len]  — 0=PUBLIC, 4=PRIVATE
    targets     : LongTensor [n_samples] — binary classification target
                  (1 if sequence contains any private token, else 0)
    """
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len), device=device)
    # Assign PRIVATE label (4) to a fraction of tokens at random
    private_mask = (torch.rand(n_samples, seq_len, device=device) < private_frac)
    state_labels = torch.where(
        private_mask,
        torch.full_like(tokens, StateLabel.PRIVATE.value),
        torch.full_like(tokens, StateLabel.PUBLIC.value),
    )
    # Target: 1 if sequence has any private token
    targets = private_mask.any(dim=-1).long()
    return tokens, state_labels, targets


def state_labels_to_vectors(
    labels: torch.Tensor,   # [B, T] — integer state labels
    state_dim: int = 8,
    n_labels:  int = len(StateLabel),
) -> torch.Tensor:
    """Convert discrete state labels to continuous state vectors.

    Uses a one-hot encoding padded to state_dim with small noise.
    In a real system, these would be learned embeddings.

    Returns Tensor [B, T, state_dim].
    """
    B, T = labels.shape
    one_hot = torch.zeros(B, T, n_labels, device=labels.device)
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1.0)
    # Pad to state_dim
    if state_dim > n_labels:
        padding = torch.zeros(B, T, state_dim - n_labels, device=labels.device)
        one_hot = torch.cat([one_hot, padding], dim=-1)
    elif state_dim < n_labels:
        one_hot = one_hot[..., :state_dim]
    # Add small noise so the network can learn from it
    one_hot += torch.randn_like(one_hot) * 0.05
    return one_hot
