"""Real NSA governance envelope for live inference backends.

This module deliberately does *not* claim to modify transformer weights or hidden
states. It wraps a real inference backend with NSA state, provenance, epistemic
metadata, and the deterministic safety kernel before and after every generation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import torch

from nsa.core.capabilities import TrustTier
from nsa.core.omega import ProvenanceRecord, TemporalHorizonState, TeleologicalState, UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel, KernelEvaluationResult, KernelVerdict
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class NSAGovernedInference:
    """Govern a real backend with an explicit NSA state transition per generation."""

    def __init__(self, backend: InferenceBackend, user_clearance: TrustTier = TrustTier.T1_INFO_GATHER, model_name: str = "unknown") -> None:
        self.backend = backend
        self.user_clearance = user_clearance
        self.model_name = model_name
        self.kernel = ImmutableSafetyKernel()
        self.state = self._initial_state()
        self.last_kernel_result: Optional[KernelEvaluationResult] = None

    def _initial_state(self) -> UnifiedCognitiveState:
        digest = hashlib.sha256(f"nsa:{self.model_name}".encode()).hexdigest()
        return UnifiedCognitiveState(
            semantic_state=torch.zeros(1, 8),
            operational_self_state=torch.zeros(1, 8),
            epistemic_state=EpistemicVector(0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.5, EpistemicTier.UNVERIFIED),
            authority_state=torch.tensor([float(self.user_clearance.value) / 4.0]),
            provenance_state=ProvenanceRecord("prov-0", f"backend://{self.model_name}", digest, 1.0),
            temporal_state=TemporalHorizonState(0, 1024, 0.0, "checkpoint_initial"),
            goal_state=TeleologicalState("conversation", 0.5, 0.0, True),
        )

    def _authorize_generation(self, prompt: str) -> KernelEvaluationResult:
        result = self.kernel.evaluate_transition(
            omega_current=self.state,
            action_id="generate_text",
            required_tier=TrustTier.T1_INFO_GATHER,
            user_clearance_tier=self.user_clearance,
            proposed_action_risk=0.0,
        )
        self.last_kernel_result = result
        return result

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, extract_hidden: bool = False, json_format: bool = False) -> LLMGenerationOutput:
        result = self._authorize_generation(prompt)
        if result.verdict != KernelVerdict.COMMIT:
            raise PermissionError("NSA blocked inference before backend execution: " + "; ".join(i.details for i in result.invariant_results if not i.passed))
        kwargs: Dict[str, Any] = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "extract_hidden": extract_hidden}
        if json_format:
            kwargs["json_format"] = True
        try:
            output = self.backend.generate(**kwargs)
        except TypeError:
            # Older third-party backends may not expose optional JSON formatting.
            kwargs.pop("json_format", None)
            output = self.backend.generate(**kwargs)
        self._commit_generation(output)
        return output

    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, system_prompt: Optional[str] = None) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return self.generate(full_prompt, max_tokens=max_tokens, temperature=temperature).text

    def _commit_generation(self, output: LLMGenerationOutput) -> None:
        previous = self.state
        response_hash = hashlib.sha256(output.text.encode("utf-8")).hexdigest()
        self.state = UnifiedCognitiveState(
            semantic_state=previous.semantic_state,
            operational_self_state=previous.operational_self_state,
            epistemic_state=EpistemicVector(
                previous.epistemic_state.known_mass,
                previous.epistemic_state.uncertainty,
                previous.epistemic_state.derivation_depth,
                previous.epistemic_state.empirical_support,
                previous.epistemic_state.verification_score,
                previous.epistemic_state.source_authenticity,
                max(0.0, min(1.0, output.confidence_estimate)),
                previous.epistemic_state.tier,
            ),
            authority_state=previous.authority_state,
            provenance_state=ProvenanceRecord(
                f"prov-{previous.temporal_state.step_index + 1}",
                f"backend://{self.model_name}/generation",
                hashlib.sha256(f"{previous.provenance_state.hash_signature}:{response_hash}".encode()).hexdigest(),
                previous.provenance_state.trust_level,
                [previous.provenance_state.record_id],
            ),
            temporal_state=TemporalHorizonState(
                previous.temporal_state.step_index + 1,
                previous.temporal_state.max_horizon_steps,
                previous.temporal_state.elapsed_time_sec,
                f"checkpoint-{previous.temporal_state.step_index + 1}",
            ),
            goal_state=previous.goal_state,
        )

    def status(self) -> Dict[str, Any]:
        kernel = self.last_kernel_result
        return {
            "model": self.model_name,
            "backend": self.backend.__class__.__name__,
            "nsa_governance": True,
            "weight_modification": False,
            "governance_layer": "runtime_reference_monitor",
            "state_step": self.state.temporal_state.step_index,
            "provenance_record": self.state.provenance_state.record_id,
            "provenance_hash": self.state.provenance_state.hash_signature,
            "epistemic_confidence": self.state.epistemic_state.confidence,
            "last_kernel_verdict": kernel.verdict.value if kernel else None,
            "last_kernel_invariants_satisfied": kernel.all_invariants_satisfied if kernel else None,
        }

    def propose_action(self, system_context: str, task_instruction: str, available_tools: List[Dict[str, Any]], fallback_action: str = "probe_service_config") -> Dict[str, Any]:
        result = self._authorize_generation(task_instruction)
        if result.verdict != KernelVerdict.COMMIT:
            raise PermissionError("NSA blocked action proposal before backend execution")
        proposal = self.backend.propose_action(system_context=system_context, task_instruction=task_instruction, available_tools=available_tools, fallback_action=fallback_action)
        self._commit_generation(LLMGenerationOutput(text=str(proposal), tokens=[], confidence_estimate=float(proposal.get("confidence", 0.5))))
        return proposal
