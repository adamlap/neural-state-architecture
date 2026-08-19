"""
nsa/epistemic.py
================
Grounded Epistemic State Algebra & Dual-Authority Governance for NSA.

Defines the decomposed Epistemic State:
    epsilon_t = (epsilon_internal, epsilon_empirical, epsilon_formal, epsilon_provenance)

And the Grounding Operator:
    G: epsilon_internal x E_external -> epsilon_grounded

Axiom (Dual-Authority Orthogonality):
    1. Operational Authority (sigma_h): "What am I authorized to do?"
    2. Epistemic Authority (epsilon_grounded): "How justified am I in believing this?"
    Neither operational clearance nor epistemic confidence can substitute for the other:
        sigma_h !-> Truth
        epsilon_grounded !-> sigma_h
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
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
class EpistemicEvidenceChannels:
    """Explicit decomposition of internal vs external evidence sources."""

    internal_estimate: float   # Neural network self-predicted confidence [0, 1]
    empirical_support: float   # Empirical validation backing [0, 1]
    formal_proof_level: float  # Symbolic / formal proof verification [0, 1]
    provenance_trust: float    # Cryptographic / trusted source authenticity [0, 1]


@dataclass
class EpistemicVector:
    """Explicit epistemic coordinates representing grounded knowledge foundation."""

    known_mass: float           # Mass of established knowledge [0, 1]
    uncertainty: float          # Residual epistemic uncertainty [0, 1]
    derivation_depth: float     # Logical deduction chain depth [0, 1]
    empirical_support: float    # Empirical validation backing [0, 1]
    verification_score: float   # Automated test/proof verification level [0, 1]
    source_authenticity: float  # Cryptographic or provenance trust [0, 1]
    confidence: float           # Overall calibrated confidence [0, 1]
    tier: EpistemicTier         # Categorical epistemic tier
    channels: Optional[EpistemicEvidenceChannels] = None

    def __post_init__(self):
        if self.channels is None:
            self.channels = EpistemicEvidenceChannels(
                internal_estimate=self.confidence,
                empirical_support=self.empirical_support,
                formal_proof_level=self.verification_score,
                provenance_trust=self.source_authenticity,
            )

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

        known = float(t[0].item())
        unc = float(t[1].item())
        depth = float(t[2].item())
        emp = float(t[3].item())
        ver = float(t[4].item())
        src = float(t[5].item())
        conf = float(t[6].item())

        return cls(
            known_mass=known,
            uncertainty=unc,
            derivation_depth=depth,
            empirical_support=emp,
            verification_score=ver,
            source_authenticity=src,
            confidence=conf,
            tier=tier,
            channels=EpistemicEvidenceChannels(
                internal_estimate=conf,
                empirical_support=emp,
                formal_proof_level=ver,
                provenance_trust=src,
            ),
        )


class GroundingOperator:
    """Formally grounds internal neural confidence against external empirical and formal evidence.

    Prevents circular self-confidence:
    G(epsilon_internal, E_external) -> epsilon_grounded
    """

    @staticmethod
    def ground(
        internal_confidence: float,
        empirical_evidence: float = 0.0,
        formal_proof: float = 0.0,
        provenance_trust: float = 0.0,
        prior_uncalibrated_allowance: float = 0.15,
    ) -> Tuple[float, EpistemicTier]:
        """Compute grounded confidence and epistemic tier.

        The model cannot assert high confidence without external empirical, formal,
        or provenance backing.
        """
        # External anchor = strongest available external justification
        external_anchor = max(empirical_evidence, formal_proof, provenance_trust)

        # Grounded confidence is strictly bounded by external evidence + small heuristic allowance
        max_allowable_confidence = min(1.0, external_anchor + prior_uncalibrated_allowance)
        grounded_conf = min(internal_confidence, max_allowable_confidence)
        grounded_conf = max(0.0, min(1.0, grounded_conf))

        # Assign tier strictly by external evidence
        if formal_proof >= 0.95:
            tier = EpistemicTier.FORMALLY_PROVEN
        elif empirical_evidence >= 0.85:
            tier = EpistemicTier.ROBUSTLY_VALIDATED
        elif empirical_evidence >= 0.65 or provenance_trust >= 0.70:
            tier = EpistemicTier.EMPIRICALLY_VALIDATED
        elif internal_confidence > 0.5 and external_anchor > 0.2:
            tier = EpistemicTier.HEURISTIC
        else:
            tier = EpistemicTier.UNVERIFIED

        return grounded_conf, tier


class DualAuthorityValidator:
    """Enforces the Dual-Authority Orthogonality Axiom:

    1. sigma_h !-> Truth (Operational clearance does not justify epistemic belief)
    2. epsilon_grounded !-> sigma_h (High confidence cannot grant operational permissions)
    """

    @staticmethod
    def assert_orthogonality(
        proposed_action_clearance: float,
        user_clearance_limit: float,
        epistemic_confidence: float,
    ) -> bool:
        """Validate that high epistemic confidence does not bypass operational clearance."""
        if proposed_action_clearance > user_clearance_limit:
            # Operational violation: Even if epistemic confidence is 1.0, action is blocked
            return False
        return True


class EpistemicGroundingEngine(nn.Module):
    """Neural & algebraic grounding engine binding state vectors to grounded epistemic states."""

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

        # Internal self-estimated confidence head (subject to external grounding)
        self.internal_confidence_head = nn.Sequential(
            nn.Linear(epistemic_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
        empirical_evidence: Optional[torch.Tensor] = None,
        formal_proof: Optional[torch.Tensor] = None,
        provenance_trust: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute continuous grounded epistemic vector for each token position."""
        inp = torch.cat((hidden, state), dim=-1)
        raw_epistemic = self.extractor(inp)

        internal_conf = self.internal_confidence_head(raw_epistemic)

        # Default external anchors if not explicitly provided
        emp = empirical_evidence if empirical_evidence is not None else raw_epistemic[..., 3:4] * 0.5
        form = formal_proof if formal_proof is not None else torch.zeros_like(internal_conf)
        prov = provenance_trust if provenance_trust is not None else raw_epistemic[..., 5:6] * 0.5

        # Grounding: external anchor bounds the internal confidence
        external_anchor = torch.maximum(torch.maximum(emp, form), prov)
        max_allowed_conf = torch.clamp(external_anchor + 0.15, max=1.0)
        grounded_conf = torch.minimum(internal_conf, max_allowed_conf)
        grounded_unc = 1.0 - grounded_conf

        # Assemble full grounded epistemic vector
        grounded_vector = torch.cat(
            (
                raw_epistemic[..., :1],   # known mass
                grounded_unc,             # uncertainty
                raw_epistemic[..., 2:3],  # derivation depth
                emp,                      # empirical support
                form,                     # formal proof
                prov,                     # provenance trust
                grounded_conf,            # grounded confidence
                raw_epistemic[..., 7:8],  # trust tier float
            ),
            dim=-1,
        )

        return {
            "epistemic_vector": grounded_vector,
            "confidence": grounded_conf,
            "internal_confidence": internal_conf,
            "grounded_confidence": grounded_conf,
            "uncertainty": grounded_unc,
            "external_anchor": external_anchor,
        }

    def compose_evidence(
        self,
        ep_a: EpistemicVector,
        ep_b: EpistemicVector,
        rule_fidelity: float = 0.95,
    ) -> EpistemicVector:
        """Compose two pieces of evidence via deductive inference rules."""
        composed_internal = min(ep_a.channels.internal_estimate, ep_b.channels.internal_estimate) * rule_fidelity
        composed_emp = (ep_a.channels.empirical_support + ep_b.channels.empirical_support) / 2.0
        composed_form = min(ep_a.channels.formal_proof_level, ep_b.channels.formal_proof_level)
        composed_prov = min(ep_a.channels.provenance_trust, ep_b.channels.provenance_trust)

        grounded_conf, derived_tier = GroundingOperator.ground(
            internal_confidence=composed_internal,
            empirical_evidence=composed_emp,
            formal_proof=composed_form,
            provenance_trust=composed_prov,
        )

        composed_unc = 1.0 - grounded_conf
        composed_depth = min(1.0, max(ep_a.derivation_depth, ep_b.derivation_depth) + 0.1)
        composed_known = (ep_a.known_mass + ep_b.known_mass) / 2.0

        return EpistemicVector(
            known_mass=composed_known,
            uncertainty=composed_unc,
            derivation_depth=composed_depth,
            empirical_support=composed_emp,
            verification_score=composed_form,
            source_authenticity=composed_prov,
            confidence=grounded_conf,
            tier=derived_tier,
            channels=EpistemicEvidenceChannels(
                internal_estimate=composed_internal,
                empirical_support=composed_emp,
                formal_proof_level=composed_form,
                provenance_trust=composed_prov,
            ),
        )


__all__ = [
    "DualAuthorityValidator",
    "EpistemicEvidenceChannels",
    "EpistemicGroundingEngine",
    "EpistemicTier",
    "EpistemicVector",
    "GroundingOperator",
]
