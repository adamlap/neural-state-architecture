"""
nsa.objectives
==============
Dual-objective training: semantic loss + state constraint loss.

The core idea is that NSA training optimises two *coupled but separate* objectives:

    L_total = L_semantic + λ × L_state

where:
    L_semantic — standard task loss (cross-entropy, MSE, etc.)
    L_state    — penalises violations of conservation laws

The two objectives are coupled through the architecture (state gates semantics)
but are computed separately, which gives us an explicit handle on the
"how is information allowed to evolve?" question independently of
"what does this represent?".

Lagrangian Formulation
----------------------
For harder constraint satisfaction, we can use a Lagrangian penalty:

    L = L_semantic + Σ_k  λ_k × [constraint_k_violated]

where λ_k are dual variables (Lagrange multipliers) updated by gradient ascent.

We implement the simpler penalty formulation (fixed λ) by default,
with an optional augmented Lagrangian mode.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import StateLattice, StateLabel, DEFAULT_LATTICE


# ---------------------------------------------------------------------------
# Semantic Loss (wrapper for standard task losses)
# ---------------------------------------------------------------------------

class SemanticLoss(nn.Module):
    """Wrapper around a standard task loss for use in the NSA dual-objective.

    Computes the prediction loss on the semantic stream output m, ignoring state.
    This is the "what does this represent?" objective.

    Parameters
    ----------
    loss_fn : str or callable
        'cross_entropy', 'mse', or a custom callable loss function.
    """

    BUILTINS = {
        "cross_entropy": F.cross_entropy,
        "mse":           F.mse_loss,
        "nll":           F.nll_loss,
    }

    def __init__(self, loss_fn: str | Callable = "cross_entropy") -> None:
        super().__init__()
        if isinstance(loss_fn, str):
            if loss_fn not in self.BUILTINS:
                raise ValueError(f"Unknown loss_fn '{loss_fn}'. Choose from {list(self.BUILTINS)}")
            self._fn = self.BUILTINS[loss_fn]
        else:
            self._fn = loss_fn

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._fn(logits, targets)


# ---------------------------------------------------------------------------
# State Constraint Loss
# ---------------------------------------------------------------------------

class StateConstraintLoss(nn.Module):
    """Penalise violations of conservation laws in the state stream.

    For each pair of tokens (i, j) in the batch, we compute the "state level"
    (as a soft expected value over the state distribution) and penalise cases
    where information flows from a higher-restriction state to a lower-restriction
    state — i.e., where private information becomes less private.

    Two variants:
        'level'   — penalise based on continuous state levels (works with any state_dim)
        'lattice' — penalise based on explicit lattice transitions (requires discrete states)

    Parameters
    ----------
    state_dim  : int   — state vector dimension
    lattice    : StateLattice — which conservation laws to enforce
    mode       : str   — 'level' or 'lattice'
    margin     : float — hinge margin for level-based penalty
    """

    MODES = ("level", "lattice")

    def __init__(
        self,
        state_dim:  int   = 8,
        lattice:    StateLattice = DEFAULT_LATTICE,
        mode:       str   = "level",
        margin:     float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.lattice   = lattice
        self.mode      = mode
        self.margin    = margin

        # Learned projection to extract a "state level" scalar from the state vector
        self.level_proj = nn.Linear(state_dim, 1, bias=False)
        nn.init.ones_(self.level_proj.weight)  # start with mean projection

    def forward(
        self,
        states_in:  torch.Tensor,   # [B, T, state_dim] — states *before* a transformation
        states_out: torch.Tensor,   # [B, T, state_dim] — states *after* a transformation
    ) -> torch.Tensor:
        """Compute the state constraint loss.

        Conservation law (monotone): the state level must not decrease.
        We penalise  max(0,  level_in - level_out - margin).

        Parameters
        ----------
        states_in  : state before transformation (per layer or overall)
        states_out : state after transformation

        Returns
        -------
        Scalar loss tensor.
        """
        if self.mode == "level":
            return self._level_loss(states_in, states_out)
        elif self.mode == "lattice":
            return self._lattice_loss(states_in, states_out)

    def _level_loss(
        self, states_in: torch.Tensor, states_out: torch.Tensor
    ) -> torch.Tensor:
        """Hinge loss penalising reduction in state level (declassification)."""
        level_in  = self.level_proj(states_in).squeeze(-1)   # [B, T]
        level_out = self.level_proj(states_out).squeeze(-1)  # [B, T]
        # Violation: level_out < level_in → information became "less restricted"
        violation = F.relu(level_in - level_out - self.margin)  # [B, T]
        return violation.mean()

    def _lattice_loss(
        self, states_in: torch.Tensor, states_out: torch.Tensor
    ) -> torch.Tensor:
        """Lattice-based penalty.

        Treats the state vector as a soft distribution over StateLabels.
        Penalises expected transitions that violate the lattice.
        """
        # Soft distribution over labels: softmax over state vector
        n_labels = len(StateLabel)
        # Project state to n_labels logits
        B, T, D = states_in.shape
        # Simple: use the first n_labels dims as logits (assume state_dim >= n_labels)
        logits_in  = states_in[..., :n_labels]
        logits_out = states_out[..., :n_labels]
        probs_in   = F.softmax(logits_in,  dim=-1)   # [B, T, n_labels]
        probs_out  = F.softmax(logits_out, dim=-1)   # [B, T, n_labels]

        # Build penalty matrix from lattice
        labels = list(StateLabel)
        penalty_matrix = torch.zeros(n_labels, n_labels, device=states_in.device)
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                if not self.lattice.is_allowed(src, dst):
                    penalty_matrix[i, j] = self.lattice.violation_penalty(src, dst)

        # Expected penalty = Σ_{src,dst} p_in[src] * p_out[dst] * penalty[src,dst]
        # [B, T, n_labels, 1] × [B, T, 1, n_labels] → [B, T, n_labels, n_labels]
        joint = probs_in.unsqueeze(-1) * probs_out.unsqueeze(-2)
        penalty = (joint * penalty_matrix).sum(dim=(-1, -2))  # [B, T]
        return penalty.mean()

    def violation_rate(
        self, states_in: torch.Tensor, states_out: torch.Tensor
    ) -> float:
        """Fraction of (batch, token) pairs with conservation law violations.

        Useful as a metric (not differentiable — used for evaluation only).
        """
        with torch.no_grad():
            level_in  = self.level_proj(states_in).squeeze(-1)
            level_out = self.level_proj(states_out).squeeze(-1)
            violations = (level_out < level_in - self.margin).float()
            return violations.mean().item()


# ---------------------------------------------------------------------------
# NSA Dual Loss
# ---------------------------------------------------------------------------

class NSALoss(nn.Module):
    """Combined dual objective: L = L_semantic + λ × L_state.

    Optionally implements an augmented Lagrangian update for λ,
    which gradually increases the constraint weight as training progresses.

    Parameters
    ----------
    semantic_loss : SemanticLoss or callable
    state_loss    : StateConstraintLoss
    lambda_init   : float — initial constraint weight λ
    lambda_max    : float — maximum λ (for augmented Lagrangian mode)
    augmented     : bool  — if True, update λ via gradient ascent
    augment_lr    : float — learning rate for λ update
    """

    def __init__(
        self,
        semantic_loss: SemanticLoss,
        state_loss:    StateConstraintLoss,
        lambda_init:   float = 0.1,
        lambda_max:    float = 10.0,
        augmented:     bool  = False,
        augment_lr:    float = 0.01,
    ) -> None:
        super().__init__()
        self.semantic_loss = semantic_loss
        self.state_loss    = state_loss
        self.lambda_max    = lambda_max
        self.augmented     = augmented
        self.augment_lr    = augment_lr

        if augmented:
            # λ as a learnable parameter (updated by gradient ascent)
            self._lambda = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer("_lambda", torch.tensor(lambda_init))

    @property
    def lam(self) -> torch.Tensor:
        return self._lambda.clamp(min=0.0, max=self.lambda_max)

    def forward(
        self,
        logits:     torch.Tensor,         # [B, *] — task predictions
        targets:    torch.Tensor,         # [B, *] — task targets
        states_in:  torch.Tensor,         # [B, T, state_dim] — state before last block
        states_out: torch.Tensor,         # [B, T, state_dim] — state after last block
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the combined loss.

        Returns
        -------
        total_loss : scalar Tensor
        metrics    : dict with individual loss values for logging
        """
        L_sem   = self.semantic_loss(logits, targets)
        L_state = self.state_loss(states_in, states_out)
        L_total = L_sem + self.lam * L_state

        metrics = {
            "loss/semantic": L_sem.item(),
            "loss/state":    L_state.item(),
            "loss/total":    L_total.item(),
            "lambda":        self.lam.item(),
        }
        return L_total, metrics

    def update_lambda(self, violation_rate: float) -> None:
        """Manually update λ based on observed violation rate (non-augmented mode).

        Increases λ if violations are high, decreases if constraints are already met.
        """
        if self.augmented:
            return  # Handled automatically via gradient ascent
        with torch.no_grad():
            if violation_rate > 0.05:    # More than 5% violations → tighten
                self._lambda.mul_(1.05)
            elif violation_rate < 0.01:  # Near-zero violations → can relax slightly
                self._lambda.mul_(0.98)
            self._lambda.clamp_(0.0, self.lambda_max)
