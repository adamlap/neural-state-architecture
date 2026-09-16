"""One-way projection of rich Omega telemetry into canonical proposals."""
from __future__ import annotations

from dataclasses import dataclass

from nsa.core.omega import UnifiedCognitiveState
from nsa.core.state import CanonicalState
from nsa.core.transition import TransitionProposal


@dataclass(frozen=True)
class OmegaFeedback:
    uncertainty: float
    confidence: float
    prediction_error: float
    provenance_trust: float
    self_state_norm: float


class OmegaFeedbackAdapter:
    """Convert neural/self-model telemetry into a governed soft-state proposal.

    This adapter has no commit capability. It deliberately exposes telemetry
    as ordinary soft/provenance data that must pass the canonical transaction
    validator and all configured policy/safety gates.
    """
    def project(self, state: CanonicalState, omega: UnifiedCognitiveState, *,
                prediction_error: float = 0.0, source: str = "omega://feedback") -> tuple[OmegaFeedback, TransitionProposal]:
        uncertainty = min(1.0, max(0.0, omega.epistemic_state.uncertainty))
        confidence = min(1.0, max(0.0, omega.epistemic_state.confidence))
        error = min(1.0, max(0.0, prediction_error))
        feedback = OmegaFeedback(
            uncertainty=uncertainty, confidence=confidence, prediction_error=error,
            provenance_trust=min(1.0, max(0.0, omega.provenance_state.trust_level)),
            self_state_norm=float(omega.operational_self_state.norm().item()),
        )
        proposal = TransitionProposal(
            action_id=f"omega-feedback-{state.step}",
            reason="project rich Omega telemetry into canonical soft state",
            soft_updates={"uncertainty": max(uncertainty, error), "confidence": confidence},
            provenance_source=source,
            evidence_id=omega.provenance_state.record_id,
            metadata={"prediction_error": error, "self_state_norm": feedback.self_state_norm},
        )
        return feedback, proposal


__all__ = ["OmegaFeedback", "OmegaFeedbackAdapter"]
