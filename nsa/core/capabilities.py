"""
NSA capability-theoretic authorization and trust hierarchy.

Capability constraints are authenticated as part of the token payload.
Verification is deliberately separated from nonce consumption so rejected
pre-execution proposals do not burn capabilities.
"""
from __future__ import annotations

import enum
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple


class TrustTier(enum.IntEnum):
    T0_COGNITION = 0
    T1_INFO_GATHER = 1
    T2_REVERSIBLE = 2
    T3_SIDE_EFFECTS = 3
    T4_CRITICAL = 4


@dataclass(frozen=True)
class CapabilityToken:
    principal: str
    action_id: str
    scope: str
    target_tier: TrustTier
    nonce: str
    expiry_timestamp: float
    signature: str
    constraints: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        now = current_time if current_time is not None else time.time()
        return now > self.expiry_timestamp


@dataclass
class TrustThermodynamicsVector:
    t_epistemic: float
    t_cognitive: float
    t_authority: float
    t_provenance: float
    t_operational: float

    def compute_max_authorized_tier(self) -> TrustTier:
        if self.t_cognitive < 0.20 or self.t_provenance < 0.50:
            return TrustTier.T0_COGNITION
        if self.t_cognitive < 0.50:
            return TrustTier.T1_INFO_GATHER
        if self.t_cognitive < 0.75:
            return TrustTier.T2_REVERSIBLE
        if self.t_authority < 0.80:
            return TrustTier.T3_SIDE_EFFECTS
        return TrustTier.T4_CRITICAL


def _canonical_constraints(constraints: Dict[str, Any]) -> str:
    return json.dumps(constraints, sort_keys=True, separators=(",", ":"), default=str)


def _payload(token: CapabilityToken) -> str:
    return ":".join((
        token.principal,
        token.action_id,
        token.scope,
        str(token.target_tier.value),
        token.nonce,
        f"{token.expiry_timestamp:.3f}",
        _canonical_constraints(token.constraints),
    ))


class CapabilityAuthority:
    """External authority capable of minting and validating capabilities.

    The default key is retained only for backwards-compatible research/tests.
    Production deployments should inject key material explicitly.
    """

    def __init__(self, master_secret_key: bytes = b"nsa-tcb-master-secret-key-3.0") -> None:
        if not master_secret_key:
            raise ValueError("master_secret_key must be non-empty")
        self._master_secret = master_secret_key
        self._consumed_nonces: Set[str] = set()

    def mint_capability(
        self, principal: str, action_id: str, scope: str, target_tier: TrustTier,
        validity_duration_sec: float = 60.0, nonce: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> CapabilityToken:
        if validity_duration_sec <= 0:
            raise ValueError("validity_duration_sec must be positive")
        nonce_val = nonce or hashlib.sha256(f"{time.time()}:{action_id}:{principal}".encode()).hexdigest()[:16]
        expiry = time.time() + validity_duration_sec
        token = CapabilityToken(principal, action_id, scope, target_tier, nonce_val, expiry, "", dict(constraints or {}))
        signature = hmac.new(self._master_secret, _payload(token).encode("utf-8"), hashlib.sha256).hexdigest()
        return CapabilityToken(principal, action_id, scope, target_tier, nonce_val, expiry, signature, token.constraints)

    def verify_capability(
        self, token: CapabilityToken, action_id: str, required_tier: TrustTier,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Validate a capability without consuming its replay nonce."""
        if token.nonce in self._consumed_nonces:
            return False, "Capability replay attack detected: nonce already consumed."
        if token.is_expired(current_time):
            return False, "Capability expired."
        if token.action_id != action_id and token.action_id != "*":
            return False, f"Capability action mismatch: grants {token.action_id}, requested {action_id}."
        if token.target_tier < required_tier:
            return False, f"Insufficient tier: capability grants {token.target_tier.name}, requires {required_tier.name}."
        expected_sig = hmac.new(self._master_secret, _payload(token).encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(token.signature, expected_sig):
            return False, "Cryptographic capability signature forgery detected."
        return True, "Capability verified successfully."

    def consume_capability(self, token: CapabilityToken) -> Tuple[bool, str]:
        """Consume a previously verified capability immediately before use."""
        if token.nonce in self._consumed_nonces:
            return False, "Capability replay attack detected: nonce already consumed."
        self._consumed_nonces.add(token.nonce)
        return True, "Capability consumed successfully."

    def verify_and_consume_capability(
        self, token: CapabilityToken, action_id: str, required_tier: TrustTier,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Backward-compatible atomic verify-and-consume helper."""
        ok, reason = self.verify_capability(token, action_id, required_tier, current_time)
        if not ok:
            return False, reason
        return self.consume_capability(token)


__all__ = ["CapabilityToken", "CapabilityAuthority", "TrustThermodynamicsVector", "TrustTier"]
