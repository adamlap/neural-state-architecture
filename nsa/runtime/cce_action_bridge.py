"""Governed output and external action execution bridge for CCE.

This module enforces NSA complete mediation between soft CCE cognitive proposals
and actual execution sinks.

Invariants:
1. Untrusted cognitive proposals never trigger real side effects directly.
2. Every proposal must pass the Immutable Safety Kernel (ISK) reference monitor.
3. Every execution or rejection is appended to an immutable audit log.
4. Hard authority is maintained by the trusted kernel, not the soft cognitive state.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from nsa.core.capabilities import TrustTier
from nsa.core.omega import UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel, KernelEvaluationResult, KernelVerdict


class ActionRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ActionProposal:
    """An untrusted action proposed by cognition or language model."""

    action_name: str
    target: str
    parameters: Dict[str, Any]
    requested_risk: float
    confidence: float
    source: str = "cce_cognition"


@dataclass(frozen=True)
class GovernedExecutionRecord:
    """An auditable record of an evaluated action proposal."""

    timestamp_utc: float
    action_name: str
    target: str
    verdict: str
    allowed: bool
    risk_score: float
    required_tier: str
    user_clearance: str
    rejection_reason: Optional[str]
    execution_result: Optional[Any]
    audit_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CCEActionBridge:
    """Evaluates and executes action proposals under the Immutable Safety Kernel."""

    def __init__(
        self,
        kernel: Optional[ImmutableSafetyKernel] = None,
        default_clearance: TrustTier = TrustTier.T1_INFO_GATHER,
    ) -> None:
        self.kernel = kernel or ImmutableSafetyKernel()
        self.default_clearance = default_clearance
        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._audit_log: List[GovernedExecutionRecord] = []

    def register_handler(
        self,
        action_name: str,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """Register an execution sink for a specific action."""
        self._action_handlers[action_name] = handler

    def evaluate_and_execute(
        self,
        proposal: ActionProposal,
        omega: UnifiedCognitiveState,
        *,
        required_tier: TrustTier = TrustTier.T1_INFO_GATHER,
        user_clearance: Optional[TrustTier] = None,
    ) -> GovernedExecutionRecord:
        """Evaluate an untrusted action proposal through ISK and execute if COMMIT."""
        clearance = user_clearance or self.default_clearance
        t_now = time.time()

        # Hard ISK evaluation
        kernel_res: KernelEvaluationResult = self.kernel.evaluate_transition(
            omega_current=omega,
            action_id=proposal.action_name,
            required_tier=required_tier,
            user_clearance_tier=clearance,
            proposed_action_risk=proposal.requested_risk,
        )

        allowed = (kernel_res.verdict == KernelVerdict.COMMIT)
        rejection_reason = None
        exec_result = None

        if allowed:
            handler = self._action_handlers.get(proposal.action_name)
            if handler is not None:
                try:
                    exec_result = handler(proposal.parameters)
                except Exception as exc:
                    exec_result = {"error": f"HandlerExecutionError: {exc}"}
            else:
                exec_result = {"status": "simulated_commit", "parameters": proposal.parameters}
        else:
            failed_invariants = [inv.details for inv in kernel_res.invariant_results if not inv.passed]
            rejection_reason = f"ISK Verdict {kernel_res.verdict.value}: " + "; ".join(failed_invariants)

        payload = f"{t_now}:{proposal.action_name}:{proposal.target}:{allowed}:{proposal.requested_risk}"
        audit_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        record = GovernedExecutionRecord(
            timestamp_utc=t_now,
            action_name=proposal.action_name,
            target=proposal.target,
            verdict=kernel_res.verdict.value,
            allowed=allowed,
            risk_score=proposal.requested_risk,
            required_tier=required_tier.name,
            user_clearance=clearance.name,
            rejection_reason=rejection_reason,
            execution_result=exec_result,
            audit_hash=audit_hash,
        )

        self._audit_log.append(record)
        return record

    @property
    def audit_log(self) -> List[GovernedExecutionRecord]:
        return list(self._audit_log)


__all__ = [
    "ActionRiskLevel",
    "ActionProposal",
    "GovernedExecutionRecord",
    "CCEActionBridge",
]
