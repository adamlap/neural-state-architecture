"""
nsa/epistemic.py
================
Epistemic State Algebra & Evidence Grounding for NSA.

Defines the epistemic state vector:
    epsilon_t = (known, uncertain, derived, empirical, verified, source, confidence)

Enforces epistemic grounding:
An action or assertion cannot be taken with high authority unless its epistemic
state satisfies the required verification threshold.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpistemicTier(enum.Enum):
    AXIOMATIC = "AXIOMATIC"
    FORMALLY_PROVEN = "FORMALLY_PROVEN"
    ROBUSTLY_VALIDATED = "ROBUSTLY_VALIDATED"
    EMPIRICALLY_VALIDATED = "EMPIRICALLY_VALIDATED"
    HEURISTIC = "HEURISTIC"
    UNVERIFIED = "UNVERIFIED"

    @property
    def trust_score(self) -> float:
        scores = {
            EpistemicTier.AXIOMATIC: 1.0,
            EpistemicTier.FORMALLY_PROVEN: 0.95,
            EpistemicTier.ROBUSTLY_VALIDATED: 0.85,
            EpistemicTier.EMPIRICALLY_VALIDATED: 0.70,
            EpistemicTier.HEURISTIC: 0.40,
            EpistemicTier.UNVERIFIED: 0.10,
        }
        return scores[self]


@dataclass
class EpistemicVector:
    """Explicit epistemic coordinates representing knowledge foundation."""

    known_mass: float         # Mass of established knowledge [0, 1]
    uncertainty: float        # Residual epistemic uncertainty [0, 1]
    derivation_depth: float   # Logical deduction chain depth [0, 1]
    empirical_support: float  # Empirical validation backing [0, 1]
    verification_score: float # Automated test/proof verification level [0, 1]
    source_authenticity: float# Cryptographic or provenance trust [0, 1]
    confidence: float         # Overall calibrated confidence [0, 1]
    tier: EpistemicTier       # Categorical epistemic tier

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.tensor(
            [
                self.known_mass,
                self.uncertainty,
                self.derivation_depth,
                self.empirical_support,
                self.verification_score,
                self.source_authenticity,
                self.confidence,
                self.tier.trust_score,
            ],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> EpistemicVector:
        t = t.view(-1)
        trust = t[7].item() if t.numel() > 7 else t[6].item()
        tier = EpistemicTier.UNVERIFIED
        for candidate in EpistemicTier:
            if abs(candidate.trust_score - trust) < 0.08:
                tier = candidate
                break

        return cls(
            known_mass=float(t[0].item()),
            uncertainty=float(t[1].item()),
            derivation_depth=float(t[2].item()),
            empirical_support=float(t[3].item()),
            verification_score=float(t[4].item()),
            source_authenticity=float(t[5].item()),
            confidence=float(t[6].item()),
            tier=tier,
        )


class EpistemicGroundingEngine(nn.Module):
    """Neural & algebraic grounding engine binding state vectors to epistemic states."""

    def __init__(self, d_model: int, state_dim: int, epistemic_dim: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.epistemic_dim = epistemic_dim

        # Epistemic feature extractor from hidden states + operational state
        self.extractor = nn.Sequential(
            nn.Linear(d_model + state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, epistemic_dim),
            nn.Sigmoid(),
        )

        # Calibrated confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(epistemic_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        provenance_trust: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute continuous epistemic vector for each token position."""
        inp = torch.cat((hidden, state), dim=-1)
        raw_epistemic = self.extractor(inp)

        # Modulate source authenticity with external provenance trust if provided
        if provenance_trust is not None:
            if provenance_trust.shape != raw_epistemic[..., 5:6].shape:
                provenance_trust = provenance_trust.expand_as(raw_epistemic[..., 5:6])
            raw_epistemic = torch.cat(
                (raw_epistemic[..., :5], provenance_trust, raw_epistemic[..., 6:]),
                dim=-1,
            )

        confidence = self.confidence_head(raw_epistemic)

        # Epistemic uncertainty = 1.0 - confidence
        uncertainty = 1.0 - confidence
        raw_epistemic = torch.cat(
            (raw_epistemic[..., :1], uncertainty, raw_epistemic[..., 2:6], confidence, raw_epistemic[..., 7:]),
            dim=-1,
        )

        return {
            "epistemic_vector": raw_epistemic,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }

    def compose_evidence(
        self,
        ep_a: EpistemicVector,
        ep_b: EpistemicVector,
        rule_fidelity: float = 0.95,
    ) -> EpistemicVector:
        """Compose two pieces of evidence via deductive inference rules."""
        composed_conf = min(ep_a.confidence, ep_b.confidence) * rule_fidelity
        composed_unc = max(ep_a.uncertainty, ep_b.uncertainty) + (1.0 - rule_fidelity) * 0.5
        composed_unc = min(1.0, composed_unc)
        composed_emp = (ep_a.empirical_support + ep_b.empirical_support) / 2.0
        composed_ver = min(ep_a.verification_score, ep_b.verification_score)
        composed_src = min(ep_a.source_authenticity, ep_b.source_authenticity)
        composed_depth = min(1.0, max(ep_a.derivation_depth, ep_b.derivation_depth) + 0.1)
        composed_known = (ep_a.known_mass + ep_b.known_mass) / 2.0

        # Tier is bounded by the weakest link
        tier_trust = min(ep_a.tier.trust_score, ep_b.tier.trust_score) * rule_fidelity
        derived_tier = EpistemicTier.UNVERIFIED
        for t in sorted(EpistemicTier, key=lambda x: x.trust_score, reverse=True):
            if tier_trust >= t.trust_score - 0.05:
                derived_tier = t
                break

        return EpistemicVector(
            known_mass=composed_known,
            uncertainty=composed_unc,
            derivation_depth=composed_depth,
            empirical_support=composed_emp,
            verification_score=composed_ver,
            source_authenticity=composed_src,
            confidence=composed_conf,
            tier=derived_tier,
        )
