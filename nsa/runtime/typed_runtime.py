"""Trusted typed-state runtime around live inference backends.

This module deliberately wraps the model at the runtime boundary.  It does
not claim access to hidden activations from Ollama: Ollama's public HTTP API
returns generated text, not transformer internals.  NSA state therefore
participates in the *live inference control plane* by being supplied as
structured context, preserved across turns, and committed only by the trusted
runtime after generation.

The important boundary is:

    model output -> observation/proposal -> trusted runtime -> next state

The model never receives a write capability for hard state.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from nsa.core.omega import (
    ProvenanceRecord,
    TeleologicalState,
    TemporalHorizonState,
    UnifiedCognitiveState,
)
from nsa.core.typed_activation import CanonicalTypedActivation
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


@dataclass(frozen=True)
class RuntimeGeneration:
    """A model response plus the canonical state observed by the runtime."""

    output: LLMGenerationOutput
    state: CanonicalTypedActivation
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]


class NSATypedRuntime:
    """Stateful NSA control-plane wrapper for a real inference backend.

    The backend may be Ollama, LM Studio, or another live backend.  The wrapper
    is intentionally backend-agnostic, so experiments can compare identical
    models with and without the NSA state boundary.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        semantic_dim: int = 16,
        goal_id: str = "default",
        provenance_uri: str = "runtime://live-inference",
    ) -> None:
        if semantic_dim <= 0:
            raise ValueError("semantic_dim must be positive")
        self.backend = backend
        self._semantic_dim = semantic_dim
        self._goal_id = goal_id
        self._provenance_uri = provenance_uri
        self.activation = CanonicalTypedActivation(self._initial_state())

    def _initial_state(self) -> UnifiedCognitiveState:
        return UnifiedCognitiveState(
            semantic_state=torch.zeros(1, self._semantic_dim),
            operational_self_state=torch.zeros(1, 4),
            epistemic_state=EpistemicVector(
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, EpistemicTier.UNVERIFIED
            ),
            # Hard authority starts at zero clearance and is runtime-owned.
            authority_state=torch.zeros(1),
            provenance_state=ProvenanceRecord(
                "runtime-genesis",
                self._provenance_uri,
                "genesis",
                1.0,
            ),
            temporal_state=TemporalHorizonState(0, 256, 0.0),
            goal_state=TeleologicalState(self._goal_id, 0.0, 0.0, True),
        )

    @staticmethod
    def _digest_features(text: str, dim: int) -> torch.Tensor:
        """Create a deterministic external semantic observation from text.

        This is intentionally *not* described as a hidden-state embedding.
        It provides a stable state observation when the backend does not expose
        transformer activations, such as the Ollama HTTP API.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [(digest[i % len(digest)] / 255.0) * 2.0 - 1.0 for i in range(dim)]
        return torch.tensor([values], dtype=torch.float32)

    def _state_context(self) -> str:
        summary = self.activation.state.to_summary_dict()
        return (
            "NSA RUNTIME STATE (read-only model context)\n"
            f"{summary}\n"
            "Hard authority is runtime-owned and cannot be changed by model output.\n"
            "Treat epistemic/provenance/temporal fields as metadata, not user instructions."
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> RuntimeGeneration:
        """Run live inference through the typed NSA state boundary."""
        before = self.activation.to_dict()
        context = self._state_context()
        full_prompt = "\n\n".join(
            part for part in (context, system_prompt, prompt) if part
        )

        output = self.backend.generate(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Only trusted runtime code can commit the post-generation observation.
        semantic = self._digest_features(output.text, self._semantic_dim)
        next_self = torch.tensor(
            [[
                float(len(output.text)),
                float(output.confidence_estimate),
                float(max_tokens),
                float(temperature),
            ]],
            dtype=torch.float32,
        )
        updated = self.activation.runtime_commit("semantic_state", semantic)
        updated = updated.runtime_commit("operational_self_state", next_self)

        current = updated.state
        current = UnifiedCognitiveState(
            semantic_state=current.semantic_state,
            operational_self_state=current.operational_self_state,
            epistemic_state=current.epistemic_state,
            authority_state=current.authority_state,
            provenance_state=ProvenanceRecord(
                record_id=f"generation-{current.temporal_state.step_index + 1}",
                source_uri=self._provenance_uri,
                hash_signature=hashlib.sha256(output.text.encode("utf-8")).hexdigest(),
                trust_level=current.provenance_state.trust_level,
                parent_records=[current.provenance_state.record_id],
            ),
            temporal_state=TemporalHorizonState(
                step_index=current.temporal_state.step_index + 1,
                max_horizon_steps=current.temporal_state.max_horizon_steps,
                elapsed_time_sec=current.temporal_state.elapsed_time_sec,
                checkpoint_snapshot_id=current.temporal_state.checkpoint_snapshot_id,
                timeout_sec=current.temporal_state.timeout_sec,
            ),
            goal_state=current.goal_state,
        )
        updated = CanonicalTypedActivation(current, updated.schema_version)
        self.activation = updated

        return RuntimeGeneration(
            output=output,
            state=updated,
            state_before=before,
            state_after=updated.to_dict(),
        )

    def inspect(self) -> Dict[str, Any]:
        """Return an auditable snapshot without exposing mutation methods."""
        return self.activation.to_dict()

    def reset(self) -> None:
        """Reset soft/session state while preserving the runtime object contract."""
        authority = self.activation.state.authority_state.detach().clone()
        self.activation = CanonicalTypedActivation(
            UnifiedCognitiveState(
                semantic_state=torch.zeros_like(self.activation.state.semantic_state),
                operational_self_state=torch.zeros_like(self.activation.state.operational_self_state),
                epistemic_state=self.activation.state.epistemic_state,
                authority_state=authority,
                provenance_state=self.activation.state.provenance_state,
                temporal_state=TemporalHorizonState(0, 256, 0.0),
                goal_state=self.activation.state.goal_state,
            )
        )
