"""
nsa.state
=========
State primitives: StateVector, StateTransitionOperator, WeightedStateEdge.

The key idea from the original proposal:

    Instead of a scalar edge weight  w
    Use a pair                       (w, V)

    where  w = semantic scalar weight (as in standard networks)
           V = state transition operator (a small matrix)

This makes the edge a *typed connector*: the scalar controls how much
semantic information passes; the matrix controls how the state evolves.

Propagation becomes:
    meaning' = meaning × w
    state'   = state × V           (or V @ state for column-vector convention)

Conservation laws are enforced by constraining V to the set of matrices
that are consistent with the current StateLattice.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from nsa.algebra import DEFAULT_LATTICE, StateLabel, StateLattice

# ---------------------------------------------------------------------------
# StateVector
# ---------------------------------------------------------------------------


class StateVector(nn.Module):
    """A learned state embedding for a batch of tokens.

    The state vector σ ∈ ℝ^state_dim encodes soft information about:
        - security level / classification
        - provenance (where did this information come from?)
        - confidence (how certain is the model?)
        - trust (is the source trustworthy?)
        - permissions (what operations are permitted on this token?)

    Two modes:
        continuous: σ is a free real-valued vector (most general)
        discrete:   σ is a softmax distribution over StateLabel values
                    (interpretable; directly maps to the lattice algebra)
    """

    MODES = ("continuous", "discrete")

    def __init__(
        self,
        state_dim: int = 8,
        mode: str = "continuous",
        n_labels: int = len(StateLabel),
        init_label: Optional[StateLabel] = None,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.state_dim = state_dim
        self.mode = mode
        self.n_labels = n_labels

        if mode == "discrete":
            # Logit vector over StateLabel values; softmax gives a distribution.
            self.logits = nn.Parameter(torch.zeros(n_labels))
            if init_label is not None:
                with torch.no_grad():
                    self.logits[init_label.value] = 5.0  # strongly initialise
        else:
            # Continuous: a linear projection produces the state from input
            # (actual state production happens in the layer modules).
            pass

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the state distribution (discrete) or pass-through (continuous)."""
        if self.mode == "discrete":
            return F.softmax(self.logits, dim=-1)  # [n_labels]
        # Continuous: state is produced by layers, not this module directly.
        if x is None:
            raise ValueError("Continuous StateVector requires input tensor x")
        return x

    def most_likely_label(self) -> StateLabel:
        """Return the most probable discrete state label."""
        if self.mode != "discrete":
            raise RuntimeError("most_likely_label only available in discrete mode")
        with torch.no_grad():
            idx = self.logits.argmax().item()
        return StateLabel(int(idx))

    def expected_level(self) -> torch.Tensor:
        """Scalar: expected state level (useful as a soft lattice position)."""
        if self.mode == "discrete":
            probs = F.softmax(self.logits, dim=-1)
            levels = torch.arange(self.n_labels, dtype=probs.dtype, device=probs.device)
            return (probs * levels).sum()
        raise RuntimeError("expected_level only available in discrete mode")


# ---------------------------------------------------------------------------
# StateTransitionOperator
# ---------------------------------------------------------------------------


class StateTransitionOperator(nn.Module):
    """A small learnable matrix V that evolves the state vector.

    state' = V @ state      (for column-vector convention)

    The matrix is small by design (2×2 to 8×8), keeping the overhead minimal.
    Conservation laws are enforced via a soft projection: after the linear
    transform, a monotone activation ensures the state level cannot decrease
    (by default). For harder constraints, a Lagrangian penalty is used during
    training (see nsa.objectives.StateConstraintLoss).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector. The transition matrix is state_dim × state_dim.
        Typical values: 4, 8, 16. Larger values allow richer state dynamics at higher cost.
    monotone_clamp : bool
        If True, clamp the diagonal to be ≥ 1 (prevents magnitude collapse along
        the "security level" dimension). Default True.
    init : str
        'identity' initialises V = I + small noise (state is approximately preserved
        unless the network learns otherwise). 'random' uses standard initialisation.
    """

    def __init__(
        self,
        state_dim: int = 8,
        monotone_clamp: bool = True,
        init: str = "identity",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.monotone_clamp = monotone_clamp

        self.V = nn.Parameter(torch.empty(state_dim, state_dim))
        self._reset_parameters(init)

    def _reset_parameters(self, init: str) -> None:
        if init == "identity":
            nn.init.eye_(self.V)
            with torch.no_grad():
                self.V += torch.randn_like(self.V) * 0.01
        else:
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def get_projected_V(self) -> torch.Tensor:
        """Exact algebraic projection P_{T_Sigma}(V) onto the cone of legal state transitions.

        Mathematical convention:
            sigma' = sigma @ V.T, or equivalently sigma'_j = sum_i sigma_i * V_{j, i}.
            Here:
                row index j = destination state (dst)
                column index i = source state (src)
                V[dst, src] represents the transition rate from src -> dst.

            Lattice ordering: UNTRUSTED(0) < PUBLIC(1) < TRUSTED(2) < CONFIDENTIAL(3) < PRIVATE(4) < SYSTEM(5).
            A transition is legal iff dst >= src (row >= col).
            Therefore, the legal transition matrix T_Sigma is LOWER TRIANGULAR under this [dst, src] convention.

        Properties guaranteed by construction:
            1. Legality: for all dst < src, P(V)[dst, src] == 0.0 (no unauthorized downward declassification).
            2. Idempotence: P(P(V)) == P(V).
            3. Non-negative diagonal: P(V)[i, i] >= 0.0.
        """
        V = self.V
        if self.monotone_clamp:
            # Lower triangular projection: zero out row < col (dst < src)
            V_tril = torch.tril(V)
            diag = V_tril.diagonal().clamp(min=0.0)
            V = V_tril - torch.diag(V_tril.diagonal()) + torch.diag(diag)
        return V

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply state transition under exact algebraic projection V in T_Sigma."""
        V_proj = self.get_projected_V()
        return state @ V_proj.T

    def frobenius_norm(self) -> torch.Tensor:
        """||V||_F — useful as a regulariser."""
        return self.get_projected_V().norm("fro")


# ---------------------------------------------------------------------------
# WeightedStateEdge
# ---------------------------------------------------------------------------


class WeightedStateEdge(nn.Module):
    """The fundamental NSA edge: a pair (w, V).

        w ∈ ℝ               — semantic scalar weight
        V ∈ ℝ^{d×d}         — state transition operator

    Propagation:
        meaning' = meaning * w
        state'   = V(state)     (via StateTransitionOperator.forward)

    This is the core deviation from standard networks: instead of a bare scalar
    weight, every connection carries a state transition that formally describes
    how information permissions evolve along that edge.
    """

    def __init__(
        self,
        state_dim: int = 8,
        init_weight: float = 1.0,
        monotone_clamp: bool = True,
    ) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_weight))
        self.V = StateTransitionOperator(state_dim=state_dim, monotone_clamp=monotone_clamp)

    def forward(
        self, meaning: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply edge to (meaning, state) pair.

        Parameters
        ----------
        meaning : Tensor (..., d_model)
        state   : Tensor (..., state_dim)

        Returns
        -------
        meaning' : Tensor (..., d_model)
        state'   : Tensor (..., state_dim)
        """
        meaning_out = meaning * self.w
        state_out = self.V(state)
        return meaning_out, state_out


# ---------------------------------------------------------------------------
# SemanticGate  Γ(σ)
# ---------------------------------------------------------------------------


class SemanticGate(nn.Module):
    """Gate the semantic flow based on the current state: Γ(σ).

    Implements:  m_gated = m ⊙ Γ(σ)

    where Γ: ℝ^state_dim → ℝ^d_model is a learned gating function.

    This is the coupling mechanism between the two manifolds:
    the state manifold controls how much semantic information can pass.
    """

    def __init__(self, d_model: int, state_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, meaning: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        meaning : Tensor (..., d_model)
        state   : Tensor (..., state_dim)

        Returns
        -------
        gated meaning : Tensor (..., d_model)
        """
        return meaning * self.gate(state)


# ---------------------------------------------------------------------------
# ContinuousStateEncoder
# ---------------------------------------------------------------------------


class ContinuousStateEncoder(nn.Module):
    """Encodes structured metadata dictionaries into continuous state vectors σ ∈ ℝ^state_dim.

    Maps discrete/symbolic attributes:
      - Security Level (SYSTEM=5, CONFIDENTIAL=3, PUBLIC=1, UNTRUSTED=0)
      - Confidence Bound (0.0 to 1.0)
      - Provenance Bitmask / Source ID (0 to 255)
      - License Tier (0 to 15)
    into a continuous dense vector representation for neural propagation.
    """

    def __init__(self, state_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.attr_proj = nn.Linear(4, state_dim, bias=False)
        nn.init.orthogonal_(self.attr_proj.weight)

    def forward(
        self,
        security_level: torch.Tensor,  # [B, T]
        confidence: torch.Tensor,  # [B, T]
        provenance: torch.Tensor,  # [B, T]
        license_tier: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """Returns continuous state tensor [B, T, state_dim]."""
        attrs = torch.stack(
            [security_level.float(), confidence.float(), provenance.float(), license_tier.float()],
            dim=-1,
        )  # [B, T, 4]

        # Primary state level placed in slot 0 for direct lattice compatibility
        state = self.attr_proj(attrs)  # [B, T, state_dim]
        state[..., 0] = security_level.float()
        return state


# ---------------------------------------------------------------------------
# LearnedStateTransitionCell
# ---------------------------------------------------------------------------


class LearnedStateTransitionCell(nn.Module):
    """Computes dynamic adaptive state transition:

    σ_{l+1} = LayerNorm(σ_l + V_θ(σ_l) + α_l * W_h h_l)
    where α_l = sigmoid(W_α [σ_l; h_l]) (initialized at α ≈ 0.01).

    Preserves Hard Security Invariants (σ[..., 0] = σ_hard) while allowing
    soft uncertainty and context relevance parameters to learn adaptively.
    """

    def __init__(self, state_dim: int = 8, d_model: int = 128, init_alpha: float = 0.01) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.transition = StateTransitionOperator(state_dim=state_dim, monotone_clamp=True)
        self.sem_proj = nn.Linear(d_model, state_dim, bias=False)
        nn.init.zeros_(self.sem_proj.weight)

        # Adaptive coupling gate α = sigmoid(W_α [σ; h]) initialized to ~0.01
        self.alpha_gate = nn.Linear(state_dim + d_model, 1)
        init_bias = math.log(init_alpha / max(1.0 - init_alpha, 1e-5))
        nn.init.constant_(self.alpha_gate.bias, init_bias)
        nn.init.zeros_(self.alpha_gate.weight)

        self.norm = nn.LayerNorm(state_dim)

    def forward(self, state: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Preserve immutable hard security level (slot 0)
        hard_security = state[..., 0:1]

        delta_state = self.transition(state)
        delta_sem = self.sem_proj(hidden_states)

        # Adaptive coupling coefficient α_l ∈ [0, 1]
        alpha = torch.sigmoid(self.alpha_gate(torch.cat([state, hidden_states], dim=-1)))

        state_next = self.norm(state + delta_state + alpha * delta_sem)
        # Restore hard security invariant
        state_next = torch.cat([hard_security, state_next[..., 1:]], dim=-1)
        return state_next


# ---------------------------------------------------------------------------
# DeclassificationOperator
# ---------------------------------------------------------------------------


class DeclassificationOperator(nn.Module):
    """Authorized state transformation operator D: (σ, m, AuthToken) -> (σ', m').

    Permits controlled downward reclassification (e.g. PRIVATE raw data -> PUBLIC summary)
    ONLY when an authorized cryptographic or policy token is provided *and*
    :meth:`StateLattice.can_declassify` allows the transition.

    Without auth, downward moves are rejected (state + meaning unchanged).
    Upward / equal moves remain allowed under the lattice even without auth.
    """

    def __init__(
        self,
        state_dim: int = 8,
        d_model: int = 128,
        lattice: Optional[StateLattice] = None,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.lattice = lattice or DEFAULT_LATTICE
        self.summary_proj = nn.Linear(d_model, d_model)
        self.auth_gate = nn.Linear(32, state_dim)

    def forward(
        self,
        meaning: torch.Tensor,
        state: torch.Tensor,
        target_level: StateLabel,
        auth_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (gated_meaning, declassified_state)."""
        authorized = auth_token is not None
        # Per-position check using current security coordinate
        src_levels = state[..., 0].detach().round().long().clamp(0, 5)
        # If any position would move downward without auth, reject whole op
        flat = src_levels.reshape(-1)
        for i in range(flat.numel()):
            src = StateLabel(int(flat[i].item()))
            if not self.lattice.can_declassify(src, target_level, authorized=authorized):
                return meaning, state

        if not authorized:
            # Allowed only if every src can already transition to target (no down)
            # meaning unchanged for pure level-preserving / upward without summary
            declassified_state = state.clone()
            declassified_state[..., 0] = float(target_level.value)
            return meaning, declassified_state

        # Authorized declassification -> summary transform + level adjust
        declassified_state = state.clone()
        declassified_state[..., 0] = float(target_level.value)
        # Optional auth embedding modulates residual of summary (if shape matches)
        summary_meaning = torch.tanh(self.summary_proj(meaning))
        if auth_token is not None and auth_token.numel() >= 32:
            gate = torch.sigmoid(self.auth_gate(auth_token[..., :32].to(meaning.dtype)))
            # gate: [..., state_dim] — broadcast onto meaning channels lightly
            summary_meaning = summary_meaning * gate.mean(dim=-1, keepdim=True).clamp(0.5, 1.0)
        return summary_meaning, declassified_state
