"""Adapters that bind CCE scheduling to the authoritative NSA substrate.

The continuous engine deliberately knows nothing about cognition or policy. This
module supplies the missing boundary for the production path: each CCE tick is
converted into exactly one ``CognitiveDynamicsSubstrate.step`` call, and only
the substrate's committed ``new_omega`` becomes the next scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from nsa.core.omega import UnifiedCognitiveState
from nsa.runtime.cognitive_substrate import CognitiveDynamicsSubstrate

CandidateAction = Tuple[str, torch.Tensor, float, float, bool]
CandidateProvider = Callable[[UnifiedCognitiveState], List[CandidateAction]]


@dataclass(frozen=True)
class SubstrateTransitionConfig:
    """Policy parameters forwarded unchanged to the authoritative substrate."""

    user_clearance_limit: float = 0.5
    target_action_risk: float = 1.0


class SubstrateTransition:
    """Turn a six-layer NSA substrate into the CCE ``step(state)`` contract.

    No state mutation or safety decision occurs here. ``CognitiveDynamicsSubstrate``
    remains the sole authority for epistemic evaluation, simulation, governance,
    immutable-kernel evaluation and commit/rollback.
    """

    def __init__(
        self,
        substrate: CognitiveDynamicsSubstrate,
        candidate_provider: CandidateProvider,
        *,
        config: SubstrateTransitionConfig | None = None,
    ) -> None:
        self.substrate = substrate
        self.candidate_provider = candidate_provider
        self.config = config or SubstrateTransitionConfig()

    def __call__(self, omega: UnifiedCognitiveState) -> UnifiedCognitiveState:
        candidates = self.candidate_provider(omega)
        if not candidates:
            raise ValueError("candidate_provider returned no actions")

        result = self.substrate.step(
            omega,
            candidates,
            user_clearance_limit=self.config.user_clearance_limit,
            target_action_risk=self.config.target_action_risk,
        )
        return result.new_omega
