"""Strict Capability Enforcement Gate for NSA."""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from nsa.capabilities.model import Capability, CapabilityAuthority
from nsa.decision import Decision, SecurityDecision


class CapabilityAccessDenied(PermissionError):
    """Raised when an action or capability invocation is denied by policy or missing authority."""
    pass


class CapabilityApprovalRequired(PermissionError):
    """Raised when an action requires human or external out-of-band approval before execution."""
    pass


@dataclass(frozen=True)
class CapabilityAuditRecord:
    """Immutable audit entry generated for every gated invocation attempt."""

    record_id: str
    timestamp_utc: float
    capability_name: str
    decision: str
    policy: str
    reason: str
    permitted: bool
    caller: str
    args_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CapabilityGate:
    """Mediation reference monitor between SecurityDecision and actual tool/capability execution."""

    def __init__(self, authority: Optional[CapabilityAuthority] = None, strict_approval: bool = True) -> None:
        self.authority = authority
        self.strict_approval = strict_approval
        self._audit_log: List[CapabilityAuditRecord] = []
        self._invocation_count: int = 0
        self._denial_count: int = 0

    @property
    def audit_log(self) -> Sequence[CapabilityAuditRecord]:
        return tuple(self._audit_log)

    @property
    def total_invocations(self) -> int:
        return self._invocation_count

    @property
    def total_denials(self) -> int:
        return self._denial_count

    def _record_audit(
        self,
        capability_name: str,
        decision: SecurityDecision,
        permitted: bool,
        caller: str,
        args_digest: str,
    ) -> CapabilityAuditRecord:
        ts = time.time()
        raw = f"{ts:.6f}:{capability_name}:{decision.decision.value}:{permitted}:{caller}:{args_digest}"
        rid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        rec = CapabilityAuditRecord(
            record_id=rid,
            timestamp_utc=ts,
            capability_name=capability_name,
            decision=decision.decision.value,
            policy=decision.policy,
            reason=decision.reason,
            permitted=permitted,
            caller=caller,
            args_digest=args_digest,
        )
        self._audit_log.append(rec)
        return rec

    def require(
        self,
        decision: SecurityDecision,
        capability_name: str,
        fn: Callable[..., Any],
        *args: Any,
        caller: str = "agent_system",
        approval_token: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute fn if and only if decision allows and capability requirements are satisfied.
        
        Guarantees:
        1. If decision is DENY -> fn is NEVER invoked (invocation count unchanged), CapabilityAccessDenied is raised.
        2. If decision is REQUIRE_APPROVAL and approval_token is None -> CapabilityApprovalRequired is raised.
        3. If decision is ALLOW -> fn is executed and result returned.
        """
        args_repr = f"args={len(args)}:kwargs={sorted(kwargs.keys())}"
        args_digest = hashlib.sha256(args_repr.encode("utf-8")).hexdigest()[:8]

        # 1. Check DENY or REDACT
        if decision.decision in {Decision.DENY, Decision.REDACT}:
            self._denial_count += 1
            self._record_audit(capability_name, decision, permitted=False, caller=caller, args_digest=args_digest)
            raise CapabilityAccessDenied(
                f"Capability '{capability_name}' access DENIED by policy '{decision.policy}': {decision.reason}"
            )

        # 2. Check REQUIRE_APPROVAL
        if decision.decision == Decision.REQUIRE_APPROVAL:
            if self.strict_approval and not approval_token:
                self._denial_count += 1
                self._record_audit(capability_name, decision, permitted=False, caller=caller, args_digest=args_digest)
                raise CapabilityApprovalRequired(
                    f"Capability '{capability_name}' REQUIRES APPROVAL under policy '{decision.policy}': {decision.reason}"
                )

        # 3. Check Authority if registered
        if self.authority is not None:
            # Check if capability is authorized
            has_cap = self.authority.allows(capability_name, subject=caller, action="execute", scope="local")
            if not has_cap:
                self._denial_count += 1
                self._record_audit(capability_name, decision, permitted=False, caller=caller, args_digest=args_digest)
                raise CapabilityAccessDenied(
                    f"CapabilityAuthority does not authorize subject '{caller}' for '{capability_name}'"
                )

        # 4. Permitted execution
        self._invocation_count += 1
        self._record_audit(capability_name, decision, permitted=True, caller=caller, args_digest=args_digest)
        return fn(*args, **kwargs)


__all__ = [
    "CapabilityAccessDenied",
    "CapabilityApprovalRequired",
    "CapabilityAuditRecord",
    "CapabilityGate",
]
