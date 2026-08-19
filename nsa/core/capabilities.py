"""
nsa/core/capabilities.py
========================
NSA 3.0 Capability-Theoretic Authorization & Trust Hierarchy Engine.

Implements:
1. Five-Tier Trust Hierarchy (T0 <= T1 <= T2 <= T3 <= T4).
2. Cryptographic Capability Object kappa:
   kappa = (principal, action, scope, resource, expiry, nonce, hmac_signature)
3. Multidimensional Trust Thermodynamics Vector:
   T = (T_epistemic, T_cognitive, T_authority, T_provenance, T_operational)
4. Non-forgeability theorem: kappa cannot be synthesized from Omega_cognitive.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class TrustTier(enum.IntEnum):
    """Five-Tier Trust Hierarchy."""
    T0_COGNITION = 0      # Pure internal cognition / thought / memory lookup
    T1_INFO_GATHER = 1    # Read-only information gathering / governed inspection
    T2_REVERSIBLE = 2     # Reversible computation / sandboxed evaluation
    T3_SIDE_EFFECTS = 3   # External side effects (network, storage writes)
    T4_CRITICAL = 4       # Irreversible / critical operations (keys, root actions)


@dataclass(frozen=True)
class CapabilityToken:
    """Cryptographic capability kappa representing unforgeable authority."""

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
    """Multidimensional trust thermodynamics vector T."""

    t_epistemic: float     # [0, 1] Justification backing
    t_cognitive: float     # [0, 1] Internal self-state health (1.0 - error)
    t_authority: float     # [0, 1] Operational clearance
    t_provenance: float    # [0, 1] Cryptographic lineage trust
    t_operational: float   # [0, 1] Effective operational clearance ceiling

    def compute_max_authorized_tier(self) -> TrustTier:
        """Derives the maximum allowed trust tier based on cognitive health."""
        if self.t_cognitive < 0.20 or self.t_provenance < 0.50:
            return TrustTier.T0_COGNITION
        elif self.t_cognitive < 0.50:
            return TrustTier.T1_INFO_GATHER
        elif self.t_cognitive < 0.75:
            return TrustTier.T2_REVERSIBLE
        elif self.t_authority < 0.80:
            return TrustTier.T3_SIDE_EFFECTS
        else:
            return TrustTier.T4_CRITICAL


class CapabilityAuthority:
    """External authority capable of minting and validating capabilities."""

    def __init__(self, master_secret_key: bytes = b"nsa-tcb-master-secret-key-3.0") -> None:
        self._master_secret = master_secret_key
        self._consumed_nonces: Set[str] = set()

    def mint_capability(
        self,
        principal: str,
        action_id: str,
        scope: str,
        target_tier: TrustTier,
        validity_duration_sec: float = 60.0,
        nonce: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> CapabilityToken:
        """Mint a cryptographically signed capability token kappa."""
        nonce_val = nonce or hashlib.sha256(f"{time.time()}:{action_id}:{principal}".encode()).hexdigest()[:16]
        expiry = time.time() + validity_duration_sec
        constraints_dict = constraints or {}

        payload = f"{principal}:{action_id}:{scope}:{target_tier.value}:{nonce_val}:{expiry:.3f}"
        signature = hmac.new(self._master_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

        return CapabilityToken(
            principal=principal,
            action_id=action_id,
            scope=scope,
            target_tier=target_tier,
            nonce=nonce_val,
            expiry_timestamp=expiry,
            signature=signature,
            constraints=constraints_dict,
        )

    def verify_and_consume_capability(
        self,
        token: CapabilityToken,
        action_id: str,
        required_tier: TrustTier,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Verify signature, expiry, tier match, and consume nonce atomically."""
        if token.nonce in self._consumed_nonces:
            return False, "Capability replay attack detected: nonce already consumed."

        if token.is_expired(current_time):
            return False, "Capability expired."

        if token.action_id != action_id and token.action_id != "*":
            return False, f"Capability action mismatch: grants {token.action_id}, requested {action_id}."

        if token.target_tier < required_tier:
            return False, f"Insufficient tier: capability grants {token.target_tier.name}, requires {required_tier.name}."

        # Verify HMAC signature
        payload = f"{token.principal}:{token.action_id}:{token.scope}:{token.target_tier.value}:{token.nonce}:{token.expiry_timestamp:.3f}"
        expected_sig = hmac.new(self._master_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(token.signature, expected_sig):
            return False, "Cryptographic capability signature forgery detected."

        # Atomic consumption
        self._consumed_nonces.add(token.nonce)
        return True, "Capability verified and consumed successfully."
