"""NSA capability-theoretic authorization and trust hierarchy."""
from __future__ import annotations
import enum
import functools
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import warnings
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
        return (current_time if current_time is not None else time.time()) > self.expiry_timestamp

@dataclass
class TrustThermodynamicsVector:
    t_epistemic: float
    t_cognitive: float
    t_authority: float
    t_provenance: float
    t_operational: float
    def compute_max_authorized_tier(self) -> TrustTier:
        if self.t_cognitive < 0.20 or self.t_provenance < 0.50: return TrustTier.T0_COGNITION
        if self.t_cognitive < 0.50: return TrustTier.T1_INFO_GATHER
        if self.t_cognitive < 0.75: return TrustTier.T2_REVERSIBLE
        if self.t_authority < 0.80: return TrustTier.T3_SIDE_EFFECTS
        return TrustTier.T4_CRITICAL

def _canonical_constraints(constraints: Dict[str, Any]) -> str:
    return json.dumps(constraints, sort_keys=True, separators=(",", ":"), default=str)

def _payload(token: CapabilityToken) -> str:
    return ":".join((token.principal, token.action_id, token.scope, str(token.target_tier.value),
                     token.nonce, f"{token.expiry_timestamp:.3f}", _canonical_constraints(token.constraints)))

def _locked(method):
    """Run an authority method under the instance lock (replay state is check-then-act)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

class CapabilityAuthority:
    """External authority capable of minting and validating capabilities.

    A capability can be *reserved* while an asynchronous effect is in flight.
    Reservation closes the concurrent-use window without consuming authority
    until the external operation has succeeded. Failed operations release the
    reservation and leave the token reusable.
    """
    DEMO_SECRET = b"nsa-tcb-master-secret-key-3.0"
    def __init__(self, master_secret_key: Optional[bytes] = None) -> None:
        if master_secret_key is None:
            warnings.warn(
                "CapabilityAuthority is using the built-in demo secret, which is public: anyone can forge "
                "tokens. Pass master_secret_key or use CapabilityAuthority.from_environment().",
                RuntimeWarning, stacklevel=2)
        self._master_secret = master_secret_key if master_secret_key is not None else self.DEMO_SECRET
        if not self._master_secret: raise ValueError("master_secret_key must be non-empty")
        # replay/reservation state is check-then-act; serialise it so a one-shot token
        # cannot be consumed twice by concurrent callers
        self._lock = threading.RLock()
        self._consumed_nonces: Set[str] = set()
        self._reserved_nonces: Dict[str, str] = {}

    @classmethod
    def ephemeral(cls) -> "CapabilityAuthority":
        """Authority with a fresh random secret: tokens are only valid inside this process."""
        return cls(secrets.token_bytes(32))

    @classmethod
    def from_environment(cls, variable: str = "NSA_CAPABILITY_MASTER_SECRET") -> "CapabilityAuthority":
        value = os.environ.get(variable)
        if not value:
            raise RuntimeError(f"missing required capability secret environment variable: {variable}")
        return cls(value.encode("utf-8"))

    def mint_capability(self, principal: str, action_id: str, scope: str, target_tier: TrustTier,
                        validity_duration_sec: float = 60.0, nonce: Optional[str] = None,
                        constraints: Optional[Dict[str, Any]] = None) -> CapabilityToken:
        if validity_duration_sec <= 0: raise ValueError("validity_duration_sec must be positive")
        nonce_val = nonce or secrets.token_hex(16)
        expiry = time.time() + validity_duration_sec
        token = CapabilityToken(principal, action_id, scope, target_tier, nonce_val, expiry, "", dict(constraints or {}))
        signature = hmac.new(self._master_secret, _payload(token).encode(), hashlib.sha256).hexdigest()
        return CapabilityToken(principal, action_id, scope, target_tier, nonce_val, expiry, signature, token.constraints)

    @_locked
    def verify_capability(self, token: CapabilityToken, action_id: str, required_tier: TrustTier,
                          current_time: Optional[float] = None) -> Tuple[bool, str]:
        if token.nonce in self._consumed_nonces: return False, "Capability replay attack detected: nonce already consumed."
        if token.nonce in self._reserved_nonces: return False, "Capability is currently reserved by another transaction."
        if token.is_expired(current_time): return False, "Capability expired."
        if token.action_id != action_id and token.action_id != "*": return False, f"Capability action mismatch: grants {token.action_id}, requested {action_id}."
        if token.target_tier < required_tier: return False, f"Insufficient tier: capability grants {token.target_tier.name}, requires {required_tier.name}."
        expected_sig = hmac.new(self._master_secret, _payload(token).encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(token.signature, expected_sig): return False, "Cryptographic capability signature forgery detected."
        return True, "Capability verified successfully."

    @_locked
    def reserve_capability(self, token: CapabilityToken, action_id: str, required_tier: TrustTier,
                           reservation_id: Optional[str] = None, current_time: Optional[float] = None) -> Tuple[bool, str, str | None]:
        ok, reason = self.verify_capability(token, action_id, required_tier, current_time)
        if not ok:
            return False, reason, None
        rid = reservation_id or hashlib.sha256(f"{time.time_ns()}:{token.nonce}:{action_id}".encode()).hexdigest()[:24]
        self._reserved_nonces[token.nonce] = rid
        return True, "Capability reserved successfully.", rid

    @_locked
    def release_capability(self, token: CapabilityToken, reservation_id: str) -> Tuple[bool, str]:
        current = self._reserved_nonces.get(token.nonce)
        if current is None:
            return False, "Capability is not reserved."
        if current != reservation_id:
            return False, "Capability reservation mismatch."
        del self._reserved_nonces[token.nonce]
        return True, "Capability reservation released successfully."

    @_locked
    def consume_reserved_capability(self, token: CapabilityToken, reservation_id: str) -> Tuple[bool, str]:
        current = self._reserved_nonces.get(token.nonce)
        if current != reservation_id:
            return False, "Capability reservation mismatch."
        if token.nonce in self._consumed_nonces:
            del self._reserved_nonces[token.nonce]
            return False, "Capability replay attack detected: nonce already consumed."
        del self._reserved_nonces[token.nonce]
        self._consumed_nonces.add(token.nonce)
        return True, "Reserved capability consumed successfully."

    @_locked
    def consume_capability(self, token: CapabilityToken) -> Tuple[bool, str]:
        if token.nonce in self._consumed_nonces: return False, "Capability replay attack detected: nonce already consumed."
        if token.nonce in self._reserved_nonces: return False, "Capability is currently reserved by another transaction."
        self._consumed_nonces.add(token.nonce)
        return True, "Capability consumed successfully."

    @_locked
    def verify_and_consume_capability(self, token: CapabilityToken, action_id: str, required_tier: TrustTier,
                                      current_time: Optional[float] = None) -> Tuple[bool, str]:
        ok, reason = self.verify_capability(token, action_id, required_tier, current_time)
        return (False, reason) if not ok else self.consume_capability(token)

__all__ = ["CapabilityToken", "CapabilityAuthority", "TrustThermodynamicsVector", "TrustTier"]
