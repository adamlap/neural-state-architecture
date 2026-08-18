"""
nsa.value_layer
===============
Value layer: the ν component of the full alignment state  h_t = (m_t, σ_t, ν_t).

The critical architectural distinction (from the AI alignment substrate analysis):

    HARD CONSTRAINTS  (σ, state algebra)   →  PERMITTED / FORBIDDEN
    SOFT VALUES        (ν, value layer)    →  PREFER AMONG PERMITTED

This separation prevents the consequentialist failure mode in which utility
maximisation overrides structural rights.  The lattice defines the *space* in
which decisions are allowed to occur; the value layer determines what the model
*should prefer* within that space.

Classes
-------
ValueAlignmentLoss
    Training-time loss combining:
      L_lm             – standard language modelling quality
      L_hard_constraint – extra penalty for predicting forbidden-range tokens
                          at CONFIDENTIAL positions (reinforces the algebraic mask
                          with a behavioural training signal)
      L_value_alignment – at injection/attack positions, train the model to predict
                          a safe-response token rather than comply with the injection

AlignmentStateProjector
    Lightweight head that projects the semantic stream m into a normative state ν:
      ν = (preference, uncertainty, utility, safety)
    ν can be used downstream for constrained decoding or logging.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ValueAlignmentLoss(nn.Module):
    """Dual-layer alignment loss: hard algebraic constraints + behavioural value training.

    L_total = L_lm + λ_hard * L_hard_constraint + λ_value * L_value_alignment

    Parameters
    ----------
    lambda_hard : float
        Weight on the hard-constraint penalty term.  Higher values more strongly
        penalise the model for placing probability mass on forbidden-range tokens
        at constrained positions.  Recommended: 3–10.
    lambda_value : float
        Weight on the value-alignment term.  Higher values more strongly push the
        model to output safe-response tokens at attack positions.  Recommended: 2–5.
    secret_lo, secret_hi : int
        Token ID range [secret_lo, secret_hi) that is classified SYSTEM-level.
        Probability mass in this range at CONFIDENTIAL positions is penalised.
    safe_token : int
        Token ID the model should output at attack positions instead of complying.
        Acts as the behavioural "refuse injection" signal.
    confidential_level : float
        State level assigned to CONFIDENTIAL positions (default 3.0).
    response_position : int
        Sequence position at which the model generates its response token.
        Value alignment is applied here for injection samples.
    """

    def __init__(
        self,
        lambda_hard: float = 5.0,
        lambda_value: float = 3.0,
        secret_lo: int = 700,
        secret_hi: int = 750,
        safe_token: int = 601,
        confidential_level: float = 3.0,
        response_position: int = 47,
    ) -> None:
        super().__init__()
        self.lambda_hard = lambda_hard
        self.lambda_value = lambda_value
        self.secret_lo = secret_lo
        self.secret_hi = secret_hi
        self.safe_token = safe_token
        self.confidential_level = confidential_level
        self.response_position = response_position

    def forward(
        self,
        logits: torch.Tensor,  # [B, T, V]
        lm_targets: torch.Tensor,  # [B, T]  standard next-token targets
        safe_targets: torch.Tensor,  # [B, T]  value-aligned override targets
        state_levels: torch.Tensor,  # [B, T]  float state levels per position
        injection_flags: torch.Tensor,  # [B]     bool, True for injection-attack samples
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, V = logits.shape
        device = logits.device

        # ── 1. Standard LM loss (all positions) ─────────────────────────────────
        l_lm = F.cross_entropy(
            logits.reshape(-1, V),
            lm_targets.reshape(-1),
            ignore_index=0,
        )

        # ── 2. Hard-constraint penalty ───────────────────────────────────────────
        # At CONFIDENTIAL (level 3) positions, penalise probability mass placed on
        # SYSTEM-range tokens.  This is a *training-time reinforcement* of the
        # algebraic mask: even without a mask at runtime the trained model would
        # avoid predicting forbidden tokens from constrained positions.
        l_hard = torch.zeros(1, device=device).squeeze()
        if self.lambda_hard > 0:
            conf_mask = state_levels == self.confidential_level  # [B, T]
            if conf_mask.any():
                conf_logits = logits[conf_mask]  # [N_conf, V]
                probs = F.softmax(conf_logits, dim=-1)
                # Sum of probability mass in the forbidden SYSTEM token range
                forbidden_mass = probs[:, self.secret_lo : self.secret_hi].sum(dim=-1)
                l_hard = forbidden_mass.mean()

        # ── 3. Value-alignment loss ──────────────────────────────────────────────
        # For injection samples at the response position: train the model to predict
        # the safe-response token (safe_token) rather than complying with the attack.
        # This is the *behavioural* dimension: the model learns to refuse, not just
        # structurally cannot access the secret.
        l_value = torch.zeros(1, device=device).squeeze()
        if self.lambda_value > 0 and injection_flags.any():
            inj_idx = torch.where(injection_flags)[0]
            rp = min(self.response_position, T - 1)
            inj_logits = logits[inj_idx, rp]  # [N_inj, V]
            inj_safe = safe_targets[inj_idx, rp]  # [N_inj]
            l_value = F.cross_entropy(inj_logits, inj_safe.long())

        l_total = l_lm + self.lambda_hard * l_hard + self.lambda_value * l_value

        return l_total, {
            "lm": l_lm.item(),
            "hard_constraint": l_hard.item(),
            "value_alignment": l_value.item(),
            "total": l_total.item(),
        }


class AlignmentStateProjector(nn.Module):
    """Projects the semantic stream m into a normative state ν.

    ν = (preference, uncertainty, utility, safety_score)

    This is the learnable ν component of  h_t = (m_t, σ_t, ν_t).
    It can be used for:
      - Monitoring value alignment during generation
      - Constrained decoding (reject actions where safety_score < threshold)
      - Logging and interpretability of model intentions

    Parameters
    ----------
    d_model : int  input dimension of the semantic stream m
    nu_dim  : int  dimension of the normative state vector ν
    """

    def __init__(self, d_model: int, nu_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, nu_dim),
        )
        # Interpret last dimension as safety score; clamp to [0, 1]
        self.safety_dim = nu_dim - 1

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        m : [B, T, d_model]  semantic stream from the NSA transformer

        Returns
        -------
        nu : [B, T, nu_dim]  normative state; last channel is safety score ∈ [0, 1]
        """
        nu = self.proj(m)
        # Clamp safety score to a valid probability
        nu_clamped = nu.clone()
        nu_clamped[..., self.safety_dim] = torch.sigmoid(nu[..., self.safety_dim])
        return nu_clamped
