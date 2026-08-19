"""
nsa/runtime/agent_runtime.py
============================
NSA 4.1 Local Cognitive Agent Runtime Substrate.

Connects an underlying local LLM backend (Ollama, llama.cpp, PyTorch)
to the NSA Cognitive State, Epistemic Governor, and Immutable Safety Kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from nsa.cognitive import NSACognitiveLM
from nsa.core.capabilities import (
    CapabilityAuthority,
    CapabilityToken,
    TrustTier,
)
from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.core.safety_kernel import (
    ImmutableSafetyKernel,
    KernelEvaluationResult,
    KernelVerdict,
)
from nsa.environment.sandboxed_world import (
    SandboxedWorldEnvironment,
    ToolDefinition,
)
from nsa.epistemic import EpistemicGroundingEngine, EpistemicTier, EpistemicVector
from nsa.governor.epistemic_governor import (
    EpistemicGovernor,
    GovernorDecision,
    GovernorVerdict,
)
from nsa.runtime.inference.base import InferenceBackend
from nsa.self_model import CounterfactualInternalSimulator


@dataclass
class AgentRuntimeStepTrace:
    step_index: int
    task_prompt: str
    llm_thought: str
    proposed_action: str
    proposed_params: Dict[str, Any]
    required_tier: TrustTier
    governor_verdict: GovernorVerdict
    kernel_verdict: KernelVerdict
    tool_execution_result: Optional[Dict[str, Any]]
    state_committed: bool
    logs: List[str]


class NSALocalRuntime:
    """End-to-end Local Cognitive Runtime managing LLM cognition and ISK governance."""

    def __init__(
        self,
        inference_backend: InferenceBackend,
        user_clearance: TrustTier = TrustTier.T1_INFO_GATHER,
        world_env: Optional[SandboxedWorldEnvironment] = None,
        model_cognitive_lm: Optional[NSACognitiveLM] = None,
    ) -> None:
        self.backend = inference_backend
        self.user_clearance = user_clearance
        self.world = world_env or SandboxedWorldEnvironment()

        d_model = 64
        state_dim = 8
        self.cognitive_lm = model_cognitive_lm or NSACognitiveLM(
            vocab_size=100, d_model=d_model, state_dim=state_dim, num_layers=2, num_heads=2, max_seq_len=16
        )
        self.epistemic_engine = EpistemicGroundingEngine(d_model=d_model, state_dim=state_dim)
        self.authority = CapabilityAuthority()
        self.safety_kernel = ImmutableSafetyKernel(capability_authority=self.authority)
        self.governor = EpistemicGovernor()

        self.tools = SandboxedWorldEnvironment.get_standard_tools()
        self.tool_map = {t.name: t for t in self.tools}

    def execute_agent_step(
        self,
        omega: UnifiedCognitiveState,
        task_instruction: str,
        supplied_capability: Optional[CapabilityToken] = None,
    ) -> Tuple[AgentRuntimeStepTrace, UnifiedCognitiveState]:
        logs: List[str] = []

        # 1. Ask LLM backend for proposal
        tool_descriptors = [{"name": t.name, "description": t.description, "tier": t.required_tier.name} for t in self.tools]
        proposal = self.backend.propose_action(
            system_context="You are an autonomous assistant operating under the NSA Governance Substrate.",
            task_instruction=task_instruction,
            available_tools=tool_descriptors,
        )

        thought = proposal.get("thought", "")
        action_name = proposal.get("action", "think")
        params = proposal.get("params", {})
        claimed_conf = float(proposal.get("confidence", 0.8))

        tool_def = self.tool_map.get(
            action_name,
            ToolDefinition(action_name, "Unknown action", TrustTier.T4_CRITICAL, 1.0, False),
        )
        required_tier = tool_def.required_tier
        logs.append(f"[LLM Proposal] Action: '{action_name}', Required Tier: {required_tier.name}")

        # 2. Epistemic Justification
        ep_res = self.epistemic_engine(omega.semantic_state, omega.operational_self_state)
        grounded_conf = ep_res["grounded_confidence"].mean().item()

        # 3. Deliberative Governor
        gov_decision = self.governor.evaluate_action(
            omega=omega,
            action_id=action_name,
            action_tensor=torch.randn(1, 8),
            action_clearance=float(required_tier.value) / 4.0,
            user_clearance=float(self.user_clearance.value) / 4.0,
            predicted_utility=0.85,
            is_irreversible=not tool_def.is_reversible,
        )
        logs.append(f"[Governor] Verdict: {gov_decision.verdict.value}")

        # 4. Immutable Safety Kernel
        if gov_decision.verdict == GovernorVerdict.DENY:
            kernel_res = KernelEvaluationResult(
                verdict=KernelVerdict.REJECT,
                all_invariants_satisfied=False,
                invariant_results=[],
            )
        else:
            kernel_res = self.safety_kernel.evaluate_transition(
                omega_current=omega,
                action_id=action_name,
                required_tier=required_tier,
                user_clearance_tier=self.user_clearance,
                proposed_action_risk=tool_def.risk_level,
                supplied_capability=supplied_capability,
            )
        logs.append(f"[ISK Reference Monitor] Verdict: {kernel_res.verdict.value}")

        # 5. Execution or Blocking
        tool_res: Optional[Dict[str, Any]] = None
        if kernel_res.verdict == KernelVerdict.COMMIT:
            tool_res = self.world.execute_tool(action_name, params)
            logs.append(f"[Execution] Tool '{action_name}' executed in sandbox: {tool_res}")
            committed = True

            new_step = omega.temporal_state.step_index + 1
            next_omega = UnifiedCognitiveState(
                semantic_state=omega.semantic_state,
                operational_self_state=omega.operational_self_state,
                epistemic_state=omega.epistemic_state,
                authority_state=omega.authority_state,
                provenance_state=ProvenanceRecord(
                    record_id=f"prov-{new_step}",
                    source_uri="internal://runtime/step",
                    hash_signature=kernel_res.committed_provenance_hash or omega.provenance_state.hash_signature,
                    trust_level=omega.provenance_state.trust_level,
                    parent_records=[omega.provenance_state.record_id],
                ),
                temporal_state=TemporalHorizonState(
                    step_index=new_step,
                    max_horizon_steps=omega.temporal_state.max_horizon_steps,
                    elapsed_time_sec=omega.temporal_state.elapsed_time_sec + 0.1,
                ),
                goal_state=omega.goal_state,
            )
        else:
            logs.append(f"[Blocked] Tool '{action_name}' blocked; 0 state change.")
            committed = False
            next_omega = omega

        trace = AgentRuntimeStepTrace(
            step_index=omega.temporal_state.step_index,
            task_prompt=task_instruction,
            llm_thought=thought,
            proposed_action=action_name,
            proposed_params=params,
            required_tier=required_tier,
            governor_verdict=gov_decision.verdict,
            kernel_verdict=kernel_res.verdict,
            tool_execution_result=tool_res,
            state_committed=committed,
            logs=logs,
        )

        return trace, next_omega
