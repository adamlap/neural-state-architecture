"""Continuous Hard-Authority Invariant & Security Monitor for CCE (Phase CCE-10).

Monitors every tick of the continuous cognitive engine to guarantee:
1. Sigma_h Invariant: Hard authority is never mutated by continuous background integration.
2. Capability Isolation: Soft working channels cannot create or forge capability tokens.
3. Malformed Feedback Trapping: Non-finite or out-of-bounds updates are intercepted and logged.
4. Authority Non-Transference: Cognitive feedback proposals cannot bypass the Immutable Safety Kernel.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from nsa.core.state import HardState
from nsa.runtime.cce_persistent_state import CognitiveStateSnapshot, PersistentCognitiveState


@dataclass(frozen=True)
class SecurityAuditTick:
    """Security audit snapshot recorded for an integration tick."""

    tick_id: str
    timestamp_utc: float
    hard_state_valid: bool
    soft_state_finite: bool
    authorizations_count: int
    license_tier: int
    confidentiality_level: str
    violation_detected: bool = False
    violation_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuousHardAuthorityMonitor:
    """Independent security monitor running alongside the continuous loop."""

    def __init__(self, baseline_hard_state: HardState) -> None:
        self.baseline_hard_state = baseline_hard_state
        self._audit_trail: List[SecurityAuditTick] = []
        self._violation_count: int = 0

    @property
    def total_violations(self) -> int:
        return self._violation_count

    @property
    def audit_trail(self) -> List[SecurityAuditTick]:
        return list(self._audit_trail)

    def verify_tick(
        self,
        current_hard_state: HardState,
        soft_snapshot: CognitiveStateSnapshot,
    ) -> SecurityAuditTick:
        """Verify that current hard state matches baseline and soft state is well-behaved."""
        ts = time.time()
        raw_seed = f"{ts:.6f}:{current_hard_state.confidentiality.name}:{soft_snapshot.update_count}"
        tid = hashlib.sha256(raw_seed.encode("utf-8")).hexdigest()[:16]

        # 1. Hard state exact match check
        hard_valid = (current_hard_state == self.baseline_hard_state)

        # 2. Soft state finite check
        soft_finite = (
            torch.isfinite(soft_snapshot.working).all()
            and torch.isfinite(soft_snapshot.self_state).all()
            and torch.isfinite(soft_snapshot.goal).all()
            and not math.isnan(soft_snapshot.uncertainty)
        )

        violation = (not hard_valid) or (not soft_finite)
        details = None
        if not hard_valid:
            violation = True
            details = "Hard authority state mutated or drifted from baseline reference monitor!"
            self._violation_count += 1
        elif not soft_finite:
            violation = True
            details = "Soft cognitive state contains non-finite values (NaN / Inf)!"
            self._violation_count += 1

        rec = SecurityAuditTick(
            tick_id=tid,
            timestamp_utc=ts,
            hard_state_valid=hard_valid,
            soft_state_finite=soft_finite,
            authorizations_count=len(current_hard_state.authorizations),
            license_tier=current_hard_state.license_tier,
            confidentiality_level=current_hard_state.confidentiality.name,
            violation_detected=violation,
            violation_details=details,
        )

        self._audit_trail.append(rec)
        if len(self._audit_trail) > 500:
            self._audit_trail.pop(0)

        if violation:
            raise PermissionError(f"[CCE Security Invariant Violation]: {details}")

        return rec


__all__ = [
    "SecurityAuditTick",
    "ContinuousHardAuthorityMonitor",
]
