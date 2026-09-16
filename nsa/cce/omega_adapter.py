"""Explicit one-way bridge between canonical CCE state and NSA Omega.

This adapter is deliberately mechanical: it maps canonical state into the
neural substrate representation but provides no path for neural tensors to
mutate hard authority. Canonical CCE remains the source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from nsa.core.omega import ProvenanceRecord, TeleologicalState, TemporalHorizonState, UnifiedCognitiveState
from nsa.core.state import CanonicalState
from nsa.epistemic import EpistemicTier, EpistemicVector


@dataclass(frozen=True)
class OmegaDimensions:
    semantic: int = 64
    operational: int = 64
    authority: int = 8


class CanonicalOmegaAdapter:
    """Map canonical CCE state to Omega without reverse authority flow."""
    def __init__(self, dimensions: OmegaDimensions = OmegaDimensions(), *, device: str | torch.device = "cpu") -> None:
        self.dimensions = dimensions
        self.device = torch.device(device)
        if min(dimensions.semantic, dimensions.operational, dimensions.authority) < 1:
            raise ValueError("Omega dimensions must be positive")

    def to_omega(self, state: CanonicalState, *, semantic_tensor: torch.Tensor | None = None,
                 operational_tensor: torch.Tensor | None = None) -> UnifiedCognitiveState:
        semantic = semantic_tensor if semantic_tensor is not None else torch.zeros(self.dimensions.semantic, device=self.device)
        operational = operational_tensor if operational_tensor is not None else torch.zeros(self.dimensions.operational, device=self.device)
        if semantic.shape[-1] != self.dimensions.semantic or operational.shape[-1] != self.dimensions.operational:
            raise ValueError("supplied tensors do not match configured Omega dimensions")
        confidence = state.soft.confidence
        tier = (EpistemicTier.EMPIRICALLY_VALIDATED if confidence >= 0.7 else
                EpistemicTier.HEURISTIC if confidence >= 0.4 else EpistemicTier.UNVERIFIED)
        epistemic = EpistemicVector(
            known_mass=max(0.0, 1.0 - state.soft.uncertainty), uncertainty=state.soft.uncertainty,
            derivation_depth=confidence, empirical_support=confidence, verification_score=confidence,
            source_authenticity=state.provenance.sources and 1.0 or 0.0, confidence=confidence, tier=tier)
        authority = torch.zeros(self.dimensions.authority, device=self.device)
        authority[0] = float(state.hard.license_tier)
        return UnifiedCognitiveState(
            semantic_state=semantic, operational_self_state=operational, epistemic_state=epistemic,
            authority_state=authority,
            provenance_state=ProvenanceRecord(
                record_id=f"canonical-{state.step}", source_uri="canonical://state",
                hash_signature=__import__("nsa.core.transition", fromlist=["state_digest"]).state_digest(state),
                trust_level=confidence, parent_records=[]),
            temporal_state=TemporalHorizonState(state.step, max_horizon_steps=64, elapsed_time_sec=0.0),
            goal_state=TeleologicalState(state.goals.active_goal or "none", state.goals.priority, state.soft.uncertainty),
        )

    def to_canonical_semantic(self, semantic_value: Any) -> Any:
        """Return model output as data; callers must submit it as a governed proposal."""
        return semantic_value


__all__ = ["CanonicalOmegaAdapter", "OmegaDimensions"]
