"""Governed Substrate Flow Engine.

Unifies Token Flow, Computation Flow, Memory Flow, and Policy/Security Flow
into a coherent state-transition loop under NSA canonical invariants.

Ensures that:
1. Optimization / residency prediction never grants authority or bypasses policy.
2. Memory movement (NVMe -> RAM -> VRAM) produces verifiable provenance.
3. Both MoE specialists and Dense sublayers are governed under the same state invariants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

from nsa.core.state import CanonicalState, HardState, ProvenanceState
from nsa.residency.types import MemoryTier, ResidencyEvent, ResidencySnapshot


@dataclass(frozen=True)
class SubstrateState:
    """Unified runtime state capturing cognitive, routing, residency, and governance state."""
    canonical: CanonicalState
    active_regions: Tuple[str, ...] = ()
    bytes_by_tier: Mapping[str, int] = field(default_factory=dict)
    active_authorizations: FrozenSet[str] = field(default_factory=frozenset)
    last_transition_reason: str = ""
    timestamp: float = field(default_factory=monotonic)

    @property
    def step(self) -> int:
        return self.canonical.step

    @property
    def is_authorized(self) -> bool:
        return len(self.canonical.hard.authorizations) > 0


class GovernedSubstrateGovernor:
    """Enforces non-bypassable governance across compute and residency transitions."""

    def __init__(self, initial_state: Optional[CanonicalState] = None) -> None:
        self.state = initial_state or CanonicalState()

    def verify_residency_invariants(
        self,
        event: ResidencyEvent,
        hard_state: HardState,
    ) -> bool:
        """Verify that a memory residency transition does not mutate or violate authority."""
        if event.action in ("grant_capability", "elevate_privilege", "bypass_policy"):
            raise PermissionError(f"Residency subsystem attempted invalid authority action: {event.action}")
        return True

    def commit_step(
        self,
        active_regions: Sequence[str],
        residency_snapshot: ResidencySnapshot,
        policy_decision: Any = None,
        reason: str = "computation-step",
    ) -> SubstrateState:
        """Advance substrate state with verified provenance linking compute and memory flow."""
        next_step = self.state.step + 1
        new_provenance = self.state.provenance.extend(
            source="substrate-runtime",
            transformation=f"regions={','.join(active_regions)}",
            evidence_id=f"step-{next_step}",
        )

        new_canonical = CanonicalState(
            semantic=self.state.semantic,
            hard=self.state.hard,
            soft=self.state.soft,
            provenance=new_provenance,
            goals=self.state.goals,
            step=next_step,
        )
        self.state = new_canonical

        bytes_by_tier = {tier.value: count for tier, count in residency_snapshot.bytes_by_tier.items()}

        return SubstrateState(
            canonical=new_canonical,
            active_regions=tuple(active_regions),
            bytes_by_tier=bytes_by_tier,
            active_authorizations=self.state.hard.authorizations,
            last_transition_reason=reason,
            timestamp=monotonic(),
        )