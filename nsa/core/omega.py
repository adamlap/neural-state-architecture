"""
nsa/core/omega.py
=================
NSA 3.0 Unified Cognitive State Vector Omega_t.

Defines the complete 7-tuple cognitive state:
    Omega_t = (m_t, sigma_t, epsilon_t, sigma_h, pi_t, tau_t, g_t)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from nsa.epistemic import EpistemicTier, EpistemicVector


@dataclass
class ProvenanceRecord:
    """Cryptographic provenance and evidence lineage pi_t."""

    record_id: str
    source_uri: str
    hash_signature: str
    trust_level: float
    parent_records: List[str] = field(default_factory=list)


@dataclass
class TemporalHorizonState:
    """Temporal and planning horizon state tau_t."""

    step_index: int
    max_horizon_steps: int
    elapsed_time_sec: float
    checkpoint_snapshot_id: Optional[str] = None
    timeout_sec: float = 60.0


@dataclass
class TeleologicalState:
    """Goal, intent, and normative priority state g_t."""

    primary_goal_id: str
    utility_expected: float
    moral_uncertainty: float
    hard_precedence_active: bool = True


@dataclass
class UnifiedCognitiveState:
    """NSA 3.0 Unified Cognitive State Omega_t."""

    semantic_state: torch.Tensor          # m_t (hidden representation)
    operational_self_state: torch.Tensor  # sigma_t (working memory & metacognitive state)
    epistemic_state: EpistemicVector      # epsilon_t (grounded justification)
    authority_state: torch.Tensor         # sigma_h (immutable operational clearance)
    provenance_state: ProvenanceRecord    # pi_t (evidence lineage)
    temporal_state: TemporalHorizonState  # tau_t (step & horizon history)
    goal_state: TeleologicalState         # g_t (normative utility & intent)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "semantic_dim": self.semantic_state.shape[-1],
            "operational_self_state_norm": float(self.operational_self_state.norm().item()),
            "epistemic_tier": self.epistemic_state.tier.value,
            "epistemic_grounded_confidence": self.epistemic_state.confidence,
            "authority_clearance_dim": self.authority_state.shape[-1],
            "provenance_record_id": self.provenance_state.record_id,
            "temporal_step": self.temporal_state.step_index,
            "primary_goal": self.goal_state.primary_goal_id,
        }
