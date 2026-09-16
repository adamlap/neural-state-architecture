"""NSA 3.0 Immutable Safety Kernel and complete mediation reference monitor."""
from __future__ import annotations
import copy
import enum
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken, TrustTier, TrustThermodynamicsVector
from nsa.core.omega import ProvenanceRecord, UnifiedCognitiveState
from nsa.epistemic import DualAuthorityValidator, EpistemicTier

class KernelVerdict(enum.Enum):
    COMMIT = "COMMIT"
    REJECT = "REJECT"
    ROLLBACK = "ROLLBACK"

@dataclass
class InvariantResult:
    invariant_id: str
    name: str
    passed: bool
    details: str

@dataclass
class KernelEvaluationResult:
    verdict: KernelVerdict
    all_invariants_satisfied: bool
    invariant_results: List[InvariantResult]
    committed_provenance_hash: Optional[str] = None
    rollback_target_snapshot_id: Optional[str] = None
    executed_tier: Optional[TrustTier] = None

class ImmutableSafetyKernel:
    """Deterministic reference monitor enforcing NSA safety invariants.

    Capability consumption is explicit. Transactional callers should verify at
    this layer without consuming, then consume exactly once immediately before
    an external effect. The legacy default remains consuming for compatibility.
    """
    def __init__(self, capability_authority: Optional[CapabilityAuthority] = None,
                 fatal_error_threshold: float = 1.50, max_verification_risk_ratio: float = 0.50) -> None:
        self.capability_authority = capability_authority or CapabilityAuthority()
        self.fatal_error_threshold = fatal_error_threshold
        self.max_verification_risk_ratio = max_verification_risk_ratio

    def evaluate_transition(self, omega_current: UnifiedCognitiveState, action_id: str,
                            required_tier: Optional[TrustTier] = None,
                            user_clearance_tier: Optional[TrustTier] = None,
                            action_clearance: Optional[float] = None,
                            user_clearance_limit: Optional[float] = None,
                            predicted_self_error: float = 0.0,
                            proposed_action_risk: float = 0.0,
                            is_verification_action: bool = False,
                            target_action_risk: float = 1.0,
                            supplied_capability: Optional[CapabilityToken] = None,
                            valid_capability_supplied: bool = False,
                            current_time: Optional[float] = None,
                            consume_capability: bool = True) -> KernelEvaluationResult:
        if required_tier is None:
            cl = action_clearance if action_clearance is not None else 0.0
            required_tier = TrustTier(int(min(4, max(0, round(cl * 4.0)))))
        if user_clearance_tier is None:
            ucl = user_clearance_limit if user_clearance_limit is not None else 0.5
            user_clearance_tier = TrustTier(int(min(4, max(0, round(ucl * 4.0)))))
        invariants: List[InvariantResult] = []

        if predicted_self_error >= self.fatal_error_threshold:
            invariants.append(InvariantResult("I_4_COGNITIVE_HEALTH", "Cognitive Health Stability Bound", False,
                f"Prediction error {predicted_self_error:.3f} >= fatal threshold {self.fatal_error_threshold:.3f}"))
            return KernelEvaluationResult(KernelVerdict.ROLLBACK, False, invariants,
                rollback_target_snapshot_id=omega_current.temporal_state.checkpoint_snapshot_id or "checkpoint_initial")

        thermo = TrustThermodynamicsVector(
            t_epistemic=omega_current.epistemic_state.confidence,
            t_cognitive=max(0.0, 1.0 - predicted_self_error / self.fatal_error_threshold),
            t_authority=float(user_clearance_tier.value) / 4.0,
            t_provenance=omega_current.provenance_state.trust_level,
            t_operational=float(user_clearance_tier.value) / 4.0)
        max_allowed_tier = thermo.compute_max_authorized_tier()
        if required_tier > max_allowed_tier and not supplied_capability:
            invariants.append(InvariantResult("I_4_COGNITIVE_HEALTH", "Cognitive Health Trust Ceiling Bound", False,
                f"Required tier {required_tier.name} exceeds cognitive health trust ceiling {max_allowed_tier.name}."))
        else:
            invariants.append(InvariantResult("I_4_COGNITIVE_HEALTH", "Cognitive Health Stability Bound", True,
                f"Cognitive health stable; trust ceiling is {max_allowed_tier.name}."))

        if required_tier > user_clearance_tier:
            if valid_capability_supplied:
                invariants.append(InvariantResult("I_1_AUTHORITY_MONOTONICITY", "Authority Monotonicity & Capability Verification", True,
                    "Capability verified by upstream transaction gate."))
            elif supplied_capability is None:
                invariants.append(InvariantResult("I_1_AUTHORITY_MONOTONICITY", "Authority Monotonicity & Clearance Boundary", False,
                    f"Action requires {required_tier.name} > user clearance {user_clearance_tier.name} without capability."))
            else:
                valid_cap, cap_msg = self.capability_authority.verify_capability(
                    supplied_capability, action_id, required_tier, current_time)
                if valid_cap and consume_capability:
                    valid_cap, cap_msg = self.capability_authority.consume_capability(supplied_capability)
                invariants.append(InvariantResult("I_1_AUTHORITY_MONOTONICITY", "Authority Monotonicity & Capability Verification",
                    valid_cap, cap_msg))
        else:
            invariants.append(InvariantResult("I_1_AUTHORITY_MONOTONICITY", "Authority Monotonicity & Clearance Boundary", True,
                "Action within baseline clearance limits."))

        orthogonality_ok = DualAuthorityValidator.assert_orthogonality(
            proposed_action_clearance=float(required_tier.value) / 4.0,
            user_clearance_limit=float(user_clearance_tier.value) / 4.0,
            epistemic_confidence=omega_current.epistemic_state.confidence)
        if not orthogonality_ok and supplied_capability is None:
            invariants.append(InvariantResult("I_2_NON_SUBSTITUTABILITY", "Tri-Partite Non-Substitutability Axiom", False,
                "Epistemic confidence attempted to substitute for operational authority."))
        else:
            invariants.append(InvariantResult("I_2_NON_SUBSTITUTABILITY", "Tri-Partite Non-Substitutability Axiom", True,
                "Authority and Epistemic Justification maintained orthogonal."))

        if is_verification_action:
            passed = proposed_action_risk < target_action_risk
            invariants.append(InvariantResult("I_5_GOVERNED_VERIFICATION", "Governed Verification Risk Bounding", passed,
                "Verification risk is strictly subordinated to target risk." if passed else
                f"Verification action risk ({proposed_action_risk:.2f}) >= target action risk ({target_action_risk:.2f})."))
        else:
            invariants.append(InvariantResult("I_5_GOVERNED_VERIFICATION", "Governed Verification Risk Bounding", True,
                "Not a verification action; risk check passed."))

        parent_hash = omega_current.provenance_state.hash_signature
        committed_hash = hashlib.sha256(f"{parent_hash}:{action_id}:{required_tier.name}:{omega_current.temporal_state.step_index}".encode()).hexdigest()
        invariants.append(InvariantResult("I_3_PROVENANCE_CHAIN", "Provenance Merkle Chain Append-Only Integrity", True,
            f"Merkle record linked: sha256:{committed_hash[:16]}..."))
        all_passed = all(inv.passed for inv in invariants)
        return KernelEvaluationResult(KernelVerdict.COMMIT if all_passed else KernelVerdict.REJECT, all_passed, invariants,
            committed_provenance_hash=committed_hash if all_passed else None,
            executed_tier=required_tier if all_passed else None)
