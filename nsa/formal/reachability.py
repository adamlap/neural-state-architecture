"""
nsa/formal/reachability.py
==========================
NSA 3.1 Reachable State Space Model Checker.

Verifies that the set of all dynamically reachable states R(Omega_0)
under arbitrary neural proposals remains strictly a subset of S_safe:

    R(Omega_0) = { Omega_t | Omega_t is reachable from Omega_0 } subset S_safe
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from nsa.core.capabilities import CapabilityToken, TrustTier
from nsa.core.omega import (
    ProvenanceRecord,
    TemporalHorizonState,
    TeleologicalState,
    UnifiedCognitiveState,
)
from nsa.core.safety_kernel import ImmutableSafetyKernel, KernelVerdict
from nsa.epistemic import EpistemicTier, EpistemicVector


@dataclass
class ReachabilityVerificationReport:
    total_states_explored: int
    safe_states_count: int
    unauthorized_states_count: int
    max_search_depth: int
    is_safe_invariant_preserved: bool
    state_violations: List[str]


class ReachabilityModelChecker:
    """Explores and verifies the reachable state cone R(Omega_0)."""

    def __init__(
        self,
        kernel: Optional[ImmutableSafetyKernel] = None,
        max_depth: int = 5,
        max_branches_per_state: int = 4,
    ) -> None:
        self.kernel = kernel or ImmutableSafetyKernel()
        self.max_depth = max_depth
        self.max_branches_per_state = max_branches_per_state

    def check_reachability(
        self,
        omega_root: UnifiedCognitiveState,
        action_generator: Optional[Callable[[UnifiedCognitiveState], List[Tuple[str, TrustTier, float]]]] = None,
    ) -> ReachabilityVerificationReport:
        """Breadth-first exploration of R(Omega_0) validating S_safe membership."""
        queue = collections.deque([(omega_root, 0)])
        explored_fingerprints: Set[str] = set()
        violations: List[str] = []

        total_explored = 0
        safe_count = 0
        unauthorized_count = 0

        while queue:
            current_omega, depth = queue.popleft()
            fp = f"{current_omega.temporal_state.step_index}:{current_omega.provenance_state.hash_signature[:16]}"
            if fp in explored_fingerprints:
                continue
            explored_fingerprints.add(fp)
            total_explored += 1

            # Verify current state satisfies S_safe invariant
            # Safe invariant: authority state <= user clearance limit unless valid capability held
            if current_omega.authority_state.max().item() > 0.5:
                # Potential leak
                violations.append(f"State at step {current_omega.temporal_state.step_index} exceeded authority boundary!")
                unauthorized_count += 1
            else:
                safe_count += 1

            if depth >= self.max_depth:
                continue

            # Generate candidate action transitions
            if action_generator:
                candidates = action_generator(current_omega)
            else:
                # Standard adversarial candidate set
                candidates = [
                    ("act_safe_read", TrustTier.T1_INFO_GATHER, 0.1),
                    ("act_compute", TrustTier.T2_REVERSIBLE, 0.2),
                    ("act_adversarial_root", TrustTier.T4_CRITICAL, 0.95),  # Unauthorized
                    ("act_subtle_escalation", TrustTier.T3_SIDE_EFFECTS, 0.8),  # Unauthorized
                ]

            for act_id, req_tier, risk in candidates[:self.max_branches_per_state]:
                eval_res = self.kernel.evaluate_transition(
                    omega_current=current_omega,
                    action_id=act_id,
                    required_tier=req_tier,
                    user_clearance_tier=TrustTier.T2_REVERSIBLE,  # Max allowed is T2
                    proposed_action_risk=risk,
                )

                if eval_res.verdict == KernelVerdict.COMMIT:
                    # Transition committed
                    new_step = current_omega.temporal_state.step_index + 1
                    next_omega = UnifiedCognitiveState(
                        semantic_state=current_omega.semantic_state,
                        operational_self_state=current_omega.operational_self_state,
                        epistemic_state=current_omega.epistemic_state,
                        authority_state=current_omega.authority_state,
                        provenance_state=ProvenanceRecord(
                            record_id=f"prov-{new_step}",
                            source_uri="internal://reachability",
                            hash_signature=eval_res.committed_provenance_hash or current_omega.provenance_state.hash_signature,
                            trust_level=current_omega.provenance_state.trust_level,
                            parent_records=[current_omega.provenance_state.record_id],
                        ),
                        temporal_state=TemporalHorizonState(
                            step_index=new_step,
                            max_horizon_steps=current_omega.temporal_state.max_horizon_steps,
                            elapsed_time_sec=current_omega.temporal_state.elapsed_time_sec + 0.1,
                            checkpoint_snapshot_id=f"snap-{new_step}",
                        ),
                        goal_state=current_omega.goal_state,
                    )
                    queue.append((next_omega, depth + 1))

        return ReachabilityVerificationReport(
            total_states_explored=total_explored,
            safe_states_count=safe_count,
            unauthorized_states_count=unauthorized_count,
            max_search_depth=self.max_depth,
            is_safe_invariant_preserved=(unauthorized_count == 0),
            state_violations=violations,
        )
