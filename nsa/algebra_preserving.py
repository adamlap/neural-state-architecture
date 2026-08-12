"""
nsa/algebra_preserving.py
=========================
Algebra-Preserving State Transitions for Typed Neural Computation (TNC).

Motivation
----------
The standard approach to learned state transitions in NSA trains an
unconstrained neural function:

    σ_{l+1} = g_θ(m_l, σ_l)

This is flexible, but there is no guarantee that the learned function
respects the algebraic invariants of Σ.  In practice, experiments
(see prototype/retrofit/native_vs_retrofit_exp.py — Model C results)
reveal a ~31.75% state monotonicity violation rate for the default
Native TNC architecture.

This module provides *algebra-preserving* state transition operators that
guarantee invariants by construction:

    σ_{l+1} = σ_l ⊔ Δ_θ(m_l, σ_l),   Δ_θ ∈ Σ

Because ⊔ (join) satisfies  a ⊔ b ≥ a  for any lattice element b ≥ ⊥,
and because Δ_θ is projected back into Σ before the join, the transition
is *structurally* monotone:

    σ_{l+1} ≥ σ_l  (lattice ordering)

No post-hoc constraint checking is needed; the algebra is satisfied by
the design of the forward pass itself.

Product Algebra Dimensions
--------------------------
Each dimension of the product state vector σ = (s, c, p, lk) gets its
own dimension-specific operator matching its mathematical structure:

  Dimension │ Invariant              │ Operator
  ──────────┼────────────────────────┼──────────────────────────────────────
  security  │ monotone ↑ (BL-model)  │ σ_s ← max(σ_s, softmax(Δ_s) index)
  confidence│ monotone ↓ (worst-case)│ σ_c ← min(σ_c, sigmoid(Δ_c))
  provenance│ set union growing      │ σ_p ← σ_p | round(sigmoid(Δ_p))
  license   │ monotone ↑ (tier)      │ σ_lk← max(σ_lk, softmax(Δ_lk) index)

For the single-dimensional scalar path (state_dim=1), only the security
monotone join is applied.

Research Hypothesis
-------------------
Replacing g_θ with algebra-preserving transitions should reduce the
31.75% monotonicity violation rate of Model C to ~0%, without the
73% PPL penalty observed in Model D (which enforces policy through
ValueAlignmentLoss applied to the semantic stream).

This corresponds to the conceptual split:

    Represent  →  σ_{l+1} = σ_l ⊔ Δ_θ(m_l, σ_l)   [Model E — this module]
    Enforce    →  policy layer / value alignment       [Model D]

If successful, Model E establishes that algebraic invariants can be
guaranteed *structurally* at low capability cost, separating the
state-representation problem from the policy-enforcement problem.

Usage
-----
    from nsa.algebra_preserving import AlgebraPreservingStateTransition

    ap = AlgebraPreservingStateTransition(d_model=128, state_dim=8)
    sigma_next = ap(m_l, sigma_l)  # guaranteed σ_{l+1} ≥ σ_l
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dimension registry — maps slot index to its algebraic structure
# ---------------------------------------------------------------------------

class DimKind:
    """Tag enum for algebra-preserving join operator selection."""
    SECURITY   = "security"    # monotone ↑ via integer-valued join
    CONFIDENCE = "confidence"  # monotone ↓ via min (conservative bound)
    PROVENANCE = "provenance"  # set union via bitwise OR approximation
    LICENSE    = "license"     # monotone ↑ via integer-valued join


# Default mapping for an 8-dimensional state vector
# (matches the product algebra described in the NSA README/whitepaper)
DEFAULT_DIM_KINDS = [
    DimKind.SECURITY,    # slot 0: primary security lattice level
    DimKind.CONFIDENCE,  # slot 1: confidence / uncertainty bound
    DimKind.PROVENANCE,  # slot 2: provenance set bit 0
    DimKind.PROVENANCE,  # slot 3: provenance set bit 1
    DimKind.PROVENANCE,  # slot 4: provenance set bit 2
    DimKind.LICENSE,     # slot 5: license restriction tier
    DimKind.SECURITY,    # slot 6: secondary security context
    DimKind.CONFIDENCE,  # slot 7: secondary confidence channel
]


class AlgebraPreservingStateTransition(nn.Module):
    """
    Algebra-preserving state transition: σ_{l+1} = σ_l ⊔ Δ_θ(m_l, σ_l).

    For each state dimension, the operator Δ_θ produces a *non-negative
    increment* (relative to the current value) that is joined with the
    current state using the dimension's lattice join:

      security / license  →  max(σ_l, proj_to_level(Δ_θ))
      confidence          →  min(σ_l, sigmoid(Δ_θ))   [monotone ↓]
      provenance          →  σ_l  |  round(sigmoid(Δ_θ))

    Args:
        d_model:    Semantic representation dimensionality.
        state_dim:  Number of state dimensions in σ.
        hidden_dim: Hidden size of the state-update MLP. Defaults to d_model//2.
        n_levels:   Number of discrete security/license levels (default: 6).
        dim_kinds:  Per-slot algebra kind list. Defaults to DEFAULT_DIM_KINDS
                    (truncated/padded to state_dim).
        dropout:    Dropout probability applied to MLP output.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        hidden_dim: Optional[int] = None,
        n_levels: int = 6,
        dim_kinds: Optional[list] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model   = d_model
        self.state_dim = state_dim
        self.n_levels  = n_levels
        self.hidden_dim = hidden_dim or (d_model // 2)

        # Resolve per-dimension algebra kind
        if dim_kinds is None:
            kinds = DEFAULT_DIM_KINDS
        else:
            kinds = list(dim_kinds)
        # Pad or truncate to state_dim
        while len(kinds) < state_dim:
            kinds.append(DimKind.SECURITY)
        self.dim_kinds: list = kinds[:state_dim]

        # MLP: (m, σ) → raw_delta  [same shape as σ]
        self.mlp = nn.Sequential(
            nn.Linear(d_model + state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, state_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise so that Δ_θ ≈ 0 at the start of training (identity-like)."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        m: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute σ_{l+1} = preserve(σ_l, Δ_θ(m_l, σ_l)).

        Args:
            m:     Semantic stream  [..., d_model]
            sigma: State vector     [..., state_dim]

        Returns:
            sigma_next: Updated state  [..., state_dim]  — guaranteed σ_{l+1} ≥ σ_l
                        under each dimension's partial order.
        """
        # Raw delta from the MLP
        inp = torch.cat([m, sigma], dim=-1)            # [..., d_model + state_dim]
        delta_raw = self.mlp(inp)                       # [..., state_dim]

        # Apply dimension-specific algebra-preserving join
        sigma_next = self._algebra_join(sigma, delta_raw)
        return sigma_next

    def _algebra_join(
        self,
        sigma: torch.Tensor,
        delta_raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-dimension algebra-preserving update.

        sigma and delta_raw have shape [..., state_dim].
        Returns sigma_next of the same shape with invariants guaranteed.
        """
        parts = []
        for i, kind in enumerate(self.dim_kinds):
            s_i = sigma[..., i : i + 1]        # [..., 1]
            d_i = delta_raw[..., i : i + 1]    # [..., 1]

            if kind == DimKind.SECURITY or kind == DimKind.LICENSE:
                # Monotone ↑: map delta to a level in [0, n_levels-1] then take max
                # sigmoid maps to (0,1); multiply by (n_levels-1) for a soft level
                delta_level = torch.sigmoid(d_i) * (self.n_levels - 1)
                # join = max(current_level, delta_level) — structurally non-decreasing
                s_next = torch.max(s_i, delta_level)
                parts.append(s_next)

            elif kind == DimKind.CONFIDENCE:
                # Monotone ↓ (conservative bound): min(current, sigmoid(delta))
                # sigmoid always in (0,1); min ensures confidence never rises
                delta_conf = torch.sigmoid(d_i)
                # Clamp existing confidence to [0,1] in case it was initialised outside
                s_clamped = torch.clamp(s_i, 0.0, 1.0)
                s_next = torch.min(s_clamped, delta_conf)
                parts.append(s_next)

            elif kind == DimKind.PROVENANCE:
                # Set union: sigmoid → Bernoulli-style soft bit; OR via max
                # round(sigmoid) ∈ {0,1}; max(σ_p, bit) = σ_p | bit for binary values
                soft_bit = torch.sigmoid(d_i)
                # Soft union: max(σ_p, soft_bit) — monotone ↑, approaches 1 once set
                s_next = torch.max(s_i, soft_bit)
                parts.append(s_next)

            else:
                # Fallback: identity (pass through unchanged)
                parts.append(s_i)

        return torch.cat(parts, dim=-1)  # [..., state_dim]

    # ------------------------------------------------------------------
    # Violation checking (for evaluation only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def count_monotonicity_violations(
        self,
        sigma_seq: torch.Tensor,
        dim_index: int = 0,
        threshold: float = 0.5,
    ) -> Tuple[int, int]:
        """
        Count monotonicity violations across a sequence of state vectors.

        Args:
            sigma_seq:  [B, T, state_dim] — full sequence of state vectors.
            dim_index:  Which state dimension to check (default: 0 = security).
            threshold:  Minimum drop to count as a violation.

        Returns:
            (n_violations, n_total_positions)
        """
        # σ[b, t, dim] should be >= σ[b, t-1, dim] for security/license dims
        s = sigma_seq[..., dim_index]           # [B, T]
        prev = s[:, :-1]                        # [B, T-1]
        curr = s[:, 1:]                         # [B, T-1]
        violations = (curr < prev - threshold).sum().item()
        total = prev.numel()
        return int(violations), int(total)
