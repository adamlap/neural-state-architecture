"""Capability-based authority for NSA actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import FrozenSet


@dataclass(frozen=True)
class Capability:
    """A narrowly scoped authorization issued outside model semantics."""

    capability_id: str
    issuer: str
    subject: str
    action: str
    scope: str
    purpose: str
    expires_at: datetime | None = None
    nonce: str | None = None

    def valid_for(self, subject: str, action: str, scope: str, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if self.subject != subject or self.action != action or self.scope != scope:
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return True


@dataclass(frozen=True)
class CapabilityAuthority:
    """Trusted issuer/validator of capabilities."""

    issuer_id: str
    capabilities: FrozenSet[Capability] = frozenset()

    def issue(self, capability: Capability) -> "CapabilityAuthority":
        if capability.issuer != self.issuer_id:
            raise PermissionError("capability issuer does not match authority")
        return CapabilityAuthority(self.issuer_id, self.capabilities | {capability})

    def allows(self, capability_id: str, subject: str, action: str, scope: str) -> bool:
        return any(
            c.capability_id == capability_id and c.valid_for(subject, action, scope)
            for c in self.capabilities
        )

    def get(self, capability_id: str) -> Capability:
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        raise PermissionError(f"unknown capability: {capability_id}")


__all__ = ["Capability", "CapabilityAuthority"]
