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

from nsa.algebra import StateLattice, DEFAULT_LATTICE, build_label_attention_mask
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

    When ``use_discrete_levels`` is True (default), security level is read from
    σ[..., 0] so discrete lattice labels map 1:1 into the mask.  Otherwise a
    learned projection is used (less secure — levels can drift).
    """
    def __init__(
        self,
        state_dim: int = 8,
        temperature: float = 1.0,
        use_discrete_levels: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_discrete_levels = use_discrete_levels
        self.level_proj = nn.Linear(state_dim, 1, bias=False)
        # Default: emphasize dim-0 (canonical security coordinate)
        with torch.no_grad():
            self.level_proj.weight.zero_()
            self.level_proj.weight[0, 0] = 1.0

    def project_level(self, s: torch.Tensor) -> torch.Tensor:
        """Project state vectors to scalar security levels [..., 1]."""
        if self.use_discrete_levels:
            return s[..., 0:1]
        return self.level_proj(s)

    def forward(self, si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
        # si: query/target state [..., state_dim]
        # sj: key/source state   [..., state_dim]
        level_target = self.project_level(si)  # [..., 1]
        level_source = self.project_level(sj)  # [..., 1]
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
    compat_mode : str   — 'dot', 'mlp', 'level', or 'lattice'
    gate_mode   : str   — 'soft' (modulate scores) or 'hard' (mask forbidden)
    alpha       : float — weight of the state compatibility term
    dropout     : float — attention dropout
    lattice     : StateLattice — used by lattice/hard level modes
    use_discrete_levels : bool — read security from σ[..., 0] for level mode
    """

    COMPAT_MODES = ("dot", "mlp", "level", "lattice")
    GATE_MODES   = ("soft", "hard")

    def __init__(
        self,
        d_model:     int   = 128,
        state_dim:   int   = 8,
        num_heads:   int   = 8,
        compat_mode: str   = "level",
        gate_mode:   str   = "hard",
        alpha:       float = 1.0,
        dropout:     float = 0.0,
        lattice:     StateLattice = DEFAULT_LATTICE,
        use_discrete_levels: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.gate_mode = gate_mode
        self.alpha     = alpha
        self.lattice   = lattice
        self.compat_mode = compat_mode
        self.use_discrete_levels = use_discrete_levels

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
        elif compat_mode in ("level", "lattice"):
            self.compat = LevelCompatibility(
                state_dim, use_discrete_levels=use_discrete_levels
            )
        else:
            raise ValueError(f"compat_mode must be one of {self.COMPAT_MODES}")

        # Output gate: Γ(σ) applied to output
        self.out_gate = SemanticGate(d_model, state_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(self.d_k)

    def _state_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Compute additive state mask [B, 1|H, T, T] from state stream."""
        B, T, _ = state.shape

        # Discrete lattice path: hard non-interference from σ[..., 0] labels
        if self.compat_mode == "lattice" or (
            self.compat_mode == "level"
            and self.gate_mode == "hard"
            and self.use_discrete_levels
        ):
            labels = state[..., 0].round().long().clamp(0, 5)
            return build_label_attention_mask(
                labels, labels, lattice=self.lattice, forbidden_value=float("-inf")
            )

        if isinstance(self.compat, LevelCompatibility):
            si = state.unsqueeze(2).expand(B, T, T, -1)
            sj = state.unsqueeze(1).expand(B, T, T, -1)
        else:
            state_proj = self.state_proj(state)  # [B, T, H]
            si = state_proj.unsqueeze(2).expand(B, T, T, -1)
            sj = state_proj.unsqueeze(1).expand(B, T, T, -1)

        g = self.compat(si, sj)  # [B, T, T, 1] or [B, T, T, H]
        if g.shape[-1] == 1:
            g = g.permute(0, 3, 1, 2)  # [B, 1, T, T]
        else:
            g = g.permute(0, 3, 1, 2)  # [B, H, T, T]

        if self.gate_mode == "soft":
            return self.alpha * torch.log(g.clamp(min=1e-8))
        # hard threshold on soft compatibility
        return torch.zeros_like(g).masked_fill(g < 0.5, float("-inf"))

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
        # State compatibility gating (uses lattice in hard/lattice modes)
        # ----------------------------------------------------------------
        state_mask = self._state_mask(state)
        scores = scores + state_mask

        # Causal / padding mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Replace NaN from all-masked rows (e.g. first token + hard blocks)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)  # [B, H, T, dk]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.W_o(out)

        # Apply semantic gate Γ(σ) — state gates what flows out
        out = self.out_gate(out, state)

        return out, state
