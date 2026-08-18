"""
nsa.verifier.automaton
======================
Security Execution Automaton (Q, Sigma_h, Sigma_s, C, delta) & Cryptographic Capability Verification.

Formal runtime security automaton governing privilege transitions and enforcing
the fundamental architectural invariant:
    "Semantic content may not manufacture hard authority."
"""

from __future__ import annotations

import hmac
import hashlib
import time
import secrets
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from nsa.algebra import StateLabel


class SecurityExecutionState(IntEnum):
    """Execution state space Q of the security automaton."""
    UNTRUSTED = 0
    PUBLIC = 1
    TRUSTED = 2
    CONFIDENTIAL = 3
    PRIVATE = 4
    SYSTEM = 5
    RECOVERY = 6
    DECLASSIFY = 7

    def to_state_label(self) -> StateLabel:
        """Map execution state to base StateLabel clearance."""
        if self.value <= 5:
            return StateLabel(self.value)
        return StateLabel.CONFIDENTIAL


@dataclass(frozen=True)
class Capability:
    """Cryptographic/Environment authorization ticket granting privilege escalation.
    
    Format: c = (issuer, subject, target, scope, purpose, expiry, nonce, signature)
    """
    issuer: str
    target_state: SecurityExecutionState
    subject: str = "agent"
    scope: str = "generation_scratchpad"
    purpose: str = "system_reasoning"
    expires_at: Optional[float] = None
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))
    max_downgrade: Optional[SecurityExecutionState] = None
    signature: Optional[str] = None

    def payload_bytes(self) -> bytes:
        """Canonical byte encoding of the capability payload for cryptographic signing."""
        exp_str = f"{self.expires_at:.6f}" if self.expires_at is not None else "none"
        downgrade_str = str(self.max_downgrade.value) if self.max_downgrade is not None else "none"
        raw = f"{self.issuer}:{self.subject}:{self.target_state.value}:{self.scope}:{self.purpose}:{exp_str}:{self.nonce}:{downgrade_str}"
        return raw.encode("utf-8")

    def is_valid(
        self,
        requested_state: SecurityExecutionState,
        current_time: Optional[float] = None,
        verifier: Optional["CapabilityVerifier"] = None,
    ) -> bool:
        """Validate expiry, target state, and optional cryptographic signature."""
        if self.expires_at is not None:
            now = current_time if current_time is not None else time.time()
            if now > self.expires_at:
                return False

        if self.target_state != requested_state:
            return False

        if verifier is not None:
            return verifier.verify(self, current_time=current_time)

        return True


class CapabilitySigner:
    """External Trusted Authority that issues and cryptographically signs capabilities."""

    def __init__(self, secret_key: bytes):
        self.secret_key = secret_key

    def issue(
        self,
        issuer: str,
        target_state: SecurityExecutionState,
        subject: str = "agent",
        scope: str = "generation_scratchpad",
        purpose: str = "system_reasoning",
        ttl_seconds: Optional[float] = 300.0,
        max_downgrade: Optional[SecurityExecutionState] = None,
    ) -> Capability:
        """Issue a cryptographically signed capability ticket."""
        now = time.time()
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        nonce = secrets.token_hex(8)

        cap_unsigned = Capability(
            issuer=issuer,
            target_state=target_state,
            subject=subject,
            scope=scope,
            purpose=purpose,
            expires_at=expires_at,
            nonce=nonce,
            max_downgrade=max_downgrade,
            signature=None,
        )

        sig = hmac.new(self.secret_key, cap_unsigned.payload_bytes(), hashlib.sha256).hexdigest()

        return Capability(
            issuer=issuer,
            target_state=target_state,
            subject=subject,
            scope=scope,
            purpose=purpose,
            expires_at=expires_at,
            nonce=nonce,
            max_downgrade=max_downgrade,
            signature=sig,
        )


class CapabilityVerifier:
    """Trusted runtime verifier executing Valid(c, sigma, sigma', t)."""

    def __init__(self, secret_key: bytes, require_signature: bool = True):
        self.secret_key = secret_key
        self.require_signature = require_signature
        self._consumed_nonces: Set[str] = set()

    def verify(self, capability: Capability, current_time: Optional[float] = None) -> bool:
        """Evaluate cryptographic signature, nonce reuse, and timestamp validity."""
        if capability.nonce in self._consumed_nonces:
            return False  # Replay attack prevention

        if capability.expires_at is not None:
            now = current_time if current_time is not None else time.time()
            if now > capability.expires_at:
                return False

        if self.require_signature:
            if not capability.signature:
                return False
            expected_sig = hmac.new(
                self.secret_key, capability.payload_bytes(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(capability.signature, expected_sig):
                return False

        return True

    def consume_nonce(self, nonce: str) -> None:
        """Mark a nonce as consumed (single-use capability semantics)."""
        self._consumed_nonces.add(nonce)


class SecurityAutomaton:
    """Deterministic Security Automaton (Q, Sigma_h, Sigma_s, C, delta).

    Prevents privilege escalation from un-authenticated semantic token emissions.
    A model emitting control tags (e.g. <|start_system_thought|>) cannot transition
    to SYSTEM clearance unless accompanied by a verified active Capability in C.
    """

    def __init__(
        self,
        initial_state: SecurityExecutionState = SecurityExecutionState.CONFIDENTIAL,
        capabilities: Optional[List[Capability]] = None,
        verifier: Optional[CapabilityVerifier] = None,
    ):
        self.current_state = initial_state
        self.verifier = verifier
        self._capabilities: Dict[str, Capability] = {}  # Indexed by nonce / ID
        if capabilities:
            for cap in capabilities:
                self.grant_capability(cap)

    def grant_capability(self, capability: Capability) -> None:
        """Register an active capability ticket in the execution environment."""
        self._capabilities[capability.nonce] = capability

    def revoke_capability(self, nonce: str) -> None:
        """Revoke a specific capability ticket."""
        self._capabilities.pop(nonce, None)

    def is_transition_authorized(
        self,
        target_state: SecurityExecutionState,
        capability: Optional[Capability] = None,
        current_time: Optional[float] = None,
    ) -> bool:
        """Evaluate predicate Authorized(c_t, q_t, q_{t+1})."""
        # De-escalation or staying in same level is always structurally safe
        if target_state.value <= self.current_state.value:
            return True

        # Recovery state is an authorized safety trap
        if target_state == SecurityExecutionState.RECOVERY:
            return True

        # Check explicitly supplied capability
        if capability is not None:
            if capability.is_valid(target_state, current_time=current_time, verifier=self.verifier):
                return True

        # Check registered environment capabilities C_t
        for cap in list(self._capabilities.values()):
            if cap.target_state == target_state:
                if cap.is_valid(target_state, current_time=current_time, verifier=self.verifier):
                    return True

        # Unauthorized attempt: blocked by automaton
        return False

    def transition(
        self,
        target_state: SecurityExecutionState,
        capability: Optional[Capability] = None,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, SecurityExecutionState]:
        """Execute state transition delta(q_t, q_target, c_t)."""
        if self.is_transition_authorized(target_state, capability, current_time=current_time):
            self.current_state = target_state
            return True, self.current_state

        # Blocked: remain in current state
        return False, self.current_state

    def snapshot(self) -> Tuple[SecurityExecutionState, Dict[str, Capability]]:
        """Snapshot internal automaton state for transactional rollback."""
        return (self.current_state, dict(self._capabilities))

    def restore(self, snap: Tuple[SecurityExecutionState, Dict[str, Capability]]) -> None:
        """Restore internal automaton state from a snapshot."""
        st, caps = snap
        self.current_state = st
        self._capabilities = dict(caps)


@dataclass
class CompleteExecutionState:
    """Formal complete execution state S_t = (X_t, K_t, V_t, sigma_t, q_t, C_t, R_t).

    Ensures that rollback restores the complete execution and security context across the entire
    neural environment atomically (Invariant: Rollback(S_{t+k}) = S_t).
    """
    token_ids: torch.Tensor
    past_key_values: Any
    state_levels: Optional[torch.Tensor]
    automaton_snapshot: Tuple[SecurityExecutionState, Dict[str, Capability]]
    router_history: List[Tuple[int, int]]
    router_buffers: Dict[int, List[int]]
    sigma_h: Optional[torch.Tensor] = None
    sigma_s: Optional[torch.Tensor] = None
