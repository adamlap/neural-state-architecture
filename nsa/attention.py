"""
nsa.attention
=============
State-aware multi-head attention.

Standard attention:
    A = softmax(QK^T / √d)

State-aware attention:
    A = softmax(QK^T / √d  ×  g(σ_i, σ_j))

where g(σ_i, σ_j) is a compatibility function between the state vectors
of the query token i and key token j.

Two gating modes:
    hard  — tokens with incompatible states are fully masked (−∞ before softmax)
    soft  — compatibility score is multiplied into the attention logit

The compatibility function can be:
    dot   — g(σ_i, σ_j) = σ_i · σ_j  (cosine-style, measures alignment)
    mlp   — g(σ_i, σ_j) = MLP([σ_i; σ_j])  (learned compatibility)
    level — g(σ_i, σ_j) based on lattice ordering of discrete state levels

This attention module is the primary locus where the two manifolds interact:
semantic information flow is *gated* by state compatibility.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, DEFAULT_LATTICE
from nsa.state import SemanticGate


# ---------------------------------------------------------------------------
# State Compatibility Functions
# ---------------------------------------------------------------------------

class DotCompatibility(nn.Module):
    """g(σ_i, σ_j) = sigmoid(σ_i · σ_j / √d_state).

    High when states are aligned, low when orthogonal/opposing.
    Differentiable and parameter-free.
    """
    def forward(self, si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
        d = si.shape[-1]
        return torch.sigmoid((si * sj).sum(-1, keepdim=True) / math.sqrt(d))


class MLPCompatibility(nn.Module):
    """g(σ_i, σ_j) = MLP([σ_i ‖ σ_j]).

    Fully learned compatibility function. More expressive but has parameters.
    """
    def __init__(self, state_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or state_dim * 2
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([si, sj], dim=-1))


class LevelCompatibility(nn.Module):
    """g(σ_i, σ_j) based on soft expected level difference.

    Target (i) is query, Source (j) is key.
    Information flows from Key (j) -> Query (i).
    Monotone rule: level_target >= level_source.
    If level_target < level_source (downward information flow),
    compatibility decays to ~0, masking forbidden information flow.
    """
    def __init__(self, state_dim: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        self.level_proj = nn.Linear(state_dim, 1, bias=False)
        nn.init.ones_(self.level_proj.weight)
        self.temperature = temperature

    def forward(self, si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
        # si: query/target state [..., state_dim]
        # sj: key/source state   [..., state_dim]
        level_target = self.level_proj(si)  # [..., 1]
        level_source = self.level_proj(sj)  # [..., 1]
        return torch.sigmoid((level_target - level_source) / self.temperature)


# ---------------------------------------------------------------------------
# State-Aware Multi-Head Attention
# ---------------------------------------------------------------------------

class StateAwareAttention(nn.Module):
    """Multi-head attention gated by state compatibility.

    Modified attention scores:
        score(i, j) = (Q_i · K_j) / √d_k  +  α × log g(σ_i, σ_j)

    where g is the compatibility function. Adding in log-space preserves
    the softmax normalisation and allows smooth gating.

    Parameters
    ----------
    d_model     : int   — model (semantic) dimension
    state_dim   : int   — state vector dimension
    num_heads   : int   — attention heads
    compat_mode : str   — 'dot', 'mlp', or 'level'
    gate_mode   : str   — 'soft' (modulate scores) or 'hard' (mask forbidden)
    alpha       : float — weight of the state compatibility term
    dropout     : float — attention dropout
    """

    COMPAT_MODES = ("dot", "mlp", "level")
    GATE_MODES   = ("soft", "hard")

    def __init__(
        self,
        d_model:     int   = 128,
        state_dim:   int   = 8,
        num_heads:   int   = 8,
        compat_mode: str   = "dot",
        gate_mode:   str   = "soft",
        alpha:       float = 1.0,
        dropout:     float = 0.0,
        lattice:     StateLattice = DEFAULT_LATTICE,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.gate_mode = gate_mode
        self.alpha     = alpha
        self.lattice   = lattice

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # State projections (project state into head-local space for efficiency)
        self.state_proj = nn.Linear(state_dim, num_heads, bias=False)

        # Compatibility function
        if compat_mode == "dot":
            self.compat = DotCompatibility()
        elif compat_mode == "mlp":
            self.compat = MLPCompatibility(state_dim)
        elif compat_mode == "level":
            self.compat = LevelCompatibility(state_dim)
        else:
            raise ValueError(f"compat_mode must be one of {self.COMPAT_MODES}")

        # Output gate: Γ(σ) applied to output
        self.out_gate = SemanticGate(d_model, state_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.d_k)

    def forward(
        self,
        x:     torch.Tensor,            # [B, T, d_model]
        state: torch.Tensor,            # [B, T, state_dim]
        mask:  Optional[torch.Tensor] = None,  # [B, 1, T, T] or [B, T, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out   : Tensor [B, T, d_model] — updated semantic representations
        state : Tensor [B, T, state_dim] — state is unchanged here (updated in layer)
        """
        B, T, _ = x.shape
        H, dk   = self.num_heads, self.d_k

        # Project queries, keys, values
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)   # [B, H, T, dk]
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)

        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale  # [B, H, T, T]

        # ----------------------------------------------------------------
        # State compatibility gating
        # ----------------------------------------------------------------
        if isinstance(self.compat, LevelCompatibility):
            si = state.unsqueeze(2).expand(B, T, T, -1)   # [B, T, T, state_dim]
            sj = state.unsqueeze(1).expand(B, T, T, -1)   # [B, T, T, state_dim]
        else:
            state_proj = self.state_proj(state)  # [B, T, H]
            si = state_proj.unsqueeze(2).expand(B, T, T, H)   # [B, T, T, H]
            sj = state_proj.unsqueeze(1).expand(B, T, T, H)   # [B, T, T, H]

        # g ∈ (0, 1): compatibility score per pair per head
        g = self.compat(si, sj)   # [B, T, T, 1] or [B, T, T, H]
        if g.shape[-1] == 1:
            g = g.permute(0, 3, 1, 2)  # [B, 1, T, T]
        else:
            g = g.permute(0, 3, 1, 2)  # [B, H, T, T]

        if self.gate_mode == "soft":
            # Add log-compatibility to scores (log-space gating)
            scores = scores + self.alpha * torch.log(g.clamp(min=1e-8))
        elif self.gate_mode == "hard":
            # Mask positions where compatibility < 0.5 (hard threshold)
            scores = scores.masked_fill(g < 0.5, float("-inf"))

        # Causal / padding mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)  # [B, H, T, dk]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)

        # Apply semantic gate Γ(σ) — state gates what flows out
        out = self.out_gate(out, state)

        return out, state
