"""Adapter from the neural six-layer substrate to canonical CCE proposals."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.omega import UnifiedCognitiveState
from nsa.core.state import CanonicalState
from nsa.runtime.cognitive_substrate import CognitiveDynamicsSubstrate, CognitiveStepResult

@dataclass(frozen=True)
class SubstrateProposal:
    action: ActionCandidate | None
    substrate_result: CognitiveStepResult
    semantic_update: object | None = None
    soft_updates: dict[str, float] | None = None

class SixLayerCanonicalAdapter:
    """Use the six-layer neural substrate as proposer, never as authority."""
    def __init__(self, substrate: CognitiveDynamicsSubstrate, *, action_payload_factory=None) -> None:
        self.substrate = substrate
        self.action_payload_factory = action_payload_factory or (lambda tensor: {"tensor": tensor})

    def propose(self, state: CanonicalState, omega: UnifiedCognitiveState,
                candidates: Sequence[ActionCandidate]) -> SubstrateProposal:
        tuples = []
        for candidate in candidates:
            payload = candidate.payload if isinstance(candidate.payload, dict) else {}
            tensor = payload.get("tensor")
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"action {candidate.action_id} requires payload['tensor'] for six-layer adaptation")
            clearance = float(payload.get("clearance", min(1.0, 1.0 - candidate.risk)))
            verification = bool(payload.get("verification", False))
            tuples.append((candidate.action_id, tensor, clearance, candidate.risk, verification))
        result = self.substrate.step(omega, tuples)
        selected = next((candidate for candidate in candidates if candidate.action_id == result.executed_action_id), None)
        if not result.transition_committed:
            selected = None
        updates = {"uncertainty": min(1.0, result.kernel_result.invariant_results[0].passed is False and state.soft.uncertainty or state.soft.uncertainty)}
        return SubstrateProposal(selected, result, soft_updates=updates)

    def __call__(self, state: CanonicalState, omega: UnifiedCognitiveState,
                 candidates: Sequence[ActionCandidate]) -> ActionCandidate | None:
        return self.propose(state, omega, candidates).action

__all__ = ["SixLayerCanonicalAdapter", "SubstrateProposal"]
