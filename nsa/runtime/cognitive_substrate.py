"""
nsa/runtime/cognitive_substrate.py
==================================
NSA 3.0 Six-Layer Constrained Cognitive Dynamics Substrate.

Orchestrates the complete 6-layer cognitive cycle:
Layer 1: Neural State (m_t, sigma_t)
Layer 2: Epistemic Justification (epsilon_t = G(eps_int, E_ext))
Layer 3: Predictive Self/World Model (Omega_hat_{t+1})
Layer 4: Deliberative Epistemic Governor (G(Omega_t, a) -> {ALLOW, VERIFY, DEFER, ESCALATE, DENY})
Layer 5: Immutable Safety Kernel (K(Omega_t, a) -> {COMMIT, REJECT, ROLLBACK})
Layer 6: Verified Execution & State Commit (Omega_{t+1})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nsa.cognitive import NSACognitiveLM
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
from nsa.epistemic import EpistemicGroundingEngine, EpistemicTier, EpistemicVector
from nsa.governor.epistemic_governor import (
    EpistemicGovernor,
    GovernorDecision,
    GovernorVerdict,
)
from nsa.self_model import ConditionedPredictiveSelfModel, CounterfactualInternalSimulator


@dataclass
class CognitiveStepResult:
    step_index: int
    governor_decision: GovernorDecision
    kernel_result: KernelEvaluationResult
    executed_action_id: str
    transition_committed: bool
    new_omega: UnifiedCognitiveState
    logs: List[str]


class CognitiveDynamicsSubstrate:
    """End-to-end 6-layer cognitive substrate orchestrating NSA 3.0."""

    def __init__(
        self,
        model: NSACognitiveLM,
        epistemic_engine: EpistemicGroundingEngine,
        safety_kernel: Optional[ImmutableSafetyKernel] = None,
        governor: Optional[EpistemicGovernor] = None,
    ) -> None:
        self.model = model
        self.epistemic_engine = epistemic_engine
        self.safety_kernel = safety_kernel or ImmutableSafetyKernel()
        self.governor = governor or EpistemicGovernor()
        self.simulator = CounterfactualInternalSimulator(
            self_model=model.self_model.conditioned_model,
            uncertainty_penalty=0.5,
        )

    def step(
        self,
        omega: UnifiedCognitiveState,
        candidate_actions: List[Tuple[str, torch.Tensor, float, float, bool]],  # (id, tensor, clearance, risk, is_verify)
        user_clearance_limit: float = 0.5,
        target_action_risk: float = 1.0,
    ) -> CognitiveStepResult:
        logs: List[str] = []

        # -------------------------------------------------------------
        # Layer 1 & 2: Neural State & Grounded Epistemic Justification
        # -------------------------------------------------------------
        ep_out = self.epistemic_engine(omega.semantic_state, omega.operational_self_state)
        grounded_conf = ep_out["grounded_confidence"].mean().item()
        logs.append(f"[Layer 1-2] Epistemic justification computed: conf={grounded_conf:.3f}")

        # -------------------------------------------------------------
        # Layer 3: Counterfactual Self/World Simulation
        # -------------------------------------------------------------
        sim_candidates = [
            (act_id, act_t, act_clearance <= user_clearance_limit)
            for act_id, act_t, act_clearance, _, _ in candidate_actions
        ]
        best_sim, all_sims = self.simulator.evaluate_candidates(
            meaning=omega.semantic_state,
            current_state=omega.operational_self_state,
            candidates=sim_candidates,
        )
        selected_act_id = best_sim.action_id if best_sim else candidate_actions[0][0]
        selected_candidate = next(c for c in candidate_actions if c[0] == selected_act_id)
        _, act_tensor, act_clearance, act_risk, is_verify = selected_candidate
        logs.append(f"[Layer 3] Simulated {len(candidate_actions)} actions; selected '{selected_act_id}'")

        # -------------------------------------------------------------
        # Layer 4: Deliberative Epistemic Governor
        # -------------------------------------------------------------
        pred_err = float((omega.operational_self_state - best_sim.predicted_state).pow(2).mean().sqrt().item()) if best_sim else 0.0
        gov_decision = self.governor.evaluate_action(
            omega=omega,
            action_id=selected_act_id,
            action_tensor=act_tensor,
            action_clearance=act_clearance,
            user_clearance=user_clearance_limit,
            predicted_utility=best_sim.score if best_sim else 0.5,
            is_irreversible=(act_risk > 0.8),
            self_state_prediction_error=pred_err,
        )
        logs.append(f"[Layer 4] Governor verdict: {gov_decision.verdict.value} ({gov_decision.rationale})")

        # -------------------------------------------------------------
        # Layer 5: Immutable Safety Kernel Invariant Check
        # -------------------------------------------------------------
        if gov_decision.verdict == GovernorVerdict.DENY:
            kernel_res = KernelEvaluationResult(
                verdict=KernelVerdict.REJECT,
                all_invariants_satisfied=False,
                invariant_results=[],
            )
        else:
            kernel_res = self.safety_kernel.evaluate_transition(
                omega_current=omega,
                action_id=selected_act_id,
                action_clearance=act_clearance,
                user_clearance_limit=user_clearance_limit,
                predicted_self_error=pred_err,
                proposed_action_risk=act_risk,
                is_verification_action=is_verify,
                target_action_risk=target_action_risk,
            )
        logs.append(f"[Layer 5] Safety Kernel verdict: {kernel_res.verdict.value}")

        # -------------------------------------------------------------
        # Layer 6: Verified Execution & State Commit
        # -------------------------------------------------------------
        if kernel_res.verdict == KernelVerdict.COMMIT:
            # Commit state update S_{t+1}
            new_step = omega.temporal_state.step_index + 1
            new_provenance = ProvenanceRecord(
                record_id=f"prov-{new_step}",
                source_uri="internal://transition/committed",
                hash_signature=kernel_res.committed_provenance_hash or omega.provenance_state.hash_signature,
                trust_level=omega.provenance_state.trust_level,
                parent_records=[omega.provenance_state.record_id],
            )
            new_temporal = TemporalHorizonState(
                step_index=new_step,
                max_horizon_steps=omega.temporal_state.max_horizon_steps,
                elapsed_time_sec=omega.temporal_state.elapsed_time_sec + 0.1,
                checkpoint_snapshot_id=f"snap-{new_step}",
            )
            new_omega = UnifiedCognitiveState(
                semantic_state=omega.semantic_state,
                operational_self_state=best_sim.predicted_state if best_sim else omega.operational_self_state,
                epistemic_state=omega.epistemic_state,
                authority_state=omega.authority_state,
                provenance_state=new_provenance,
                temporal_state=new_temporal,
                goal_state=omega.goal_state,
            )
            logs.append(f"[Layer 6] State committed to step {new_step} (Merkle hash {new_provenance.hash_signature[:12]}...)")
            committed = True
        else:
            new_omega = omega
            logs.append("[Layer 6] Transition rejected / rollbacked; state unchanged.")
            committed = False

        return CognitiveStepResult(
            step_index=omega.temporal_state.step_index,
            governor_decision=gov_decision,
            kernel_result=kernel_res,
            executed_action_id=selected_act_id,
            transition_committed=committed,
            new_omega=new_omega,
            logs=logs,
        )
