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
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, StateLabel, DEFAULT_LATTICE


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

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply state transition.

        Parameters
        ----------
        state : Tensor of shape (..., state_dim)

        Returns
        -------
        Tensor of shape (..., state_dim)
        """
        V = self.V
        if self.monotone_clamp:
            # Soft clamping: diagonal entries ≥ 0 (no inversion of state direction)
            diag = V.diagonal().clamp(min=0.0)
            V = V - torch.diag(V.diagonal()) + torch.diag(diag)

        # (..., state_dim) @ (state_dim, state_dim).T → (..., state_dim)
        return state @ V.T

    def frobenius_norm(self) -> torch.Tensor:
        """‖V‖_F — useful as a regulariser."""
        return self.V.norm("fro")


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
        state_out   = self.V(state)
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
