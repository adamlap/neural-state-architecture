"""Unified coordinator for NSA token, computation, residency, and governance flow.

This module is deliberately orchestration-only: it does not replace the
canonical state transition machinery or grant authority. It joins existing
predictors, residency planning, counterfactual evaluation, memory
consolidation, and substrate governance into one observable transition loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping, Optional, Sequence

from nsa.core.state import CanonicalState
from nsa.core.substrate_governance import GovernedSubstrateGovernor, SubstrateState
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyDecision, ResidencyPolicy
from nsa.residency.types import MemoryTier, ResidencySnapshot


@dataclass(frozen=True)
class TokenEnvelope:
    token_id: int | str
    cognitive_state: Mapping[str, object] = field(default_factory=dict)
    provenance: str = ""
    authority: tuple[str, ...] = ()
    priority: float = 1.0


@dataclass(frozen=True)
class ComputationCandidate:
    region_id: str
    probability: float
    utility: float = 1.0
    latency_ms: float = 0.0
    bytes_required: int = 0
    reason: str = "prediction"


@dataclass(frozen=True)
class ResourceAllocation:
    region_id: str
    tier: MemoryTier
    precision: str
    probability: float
    expected_utility: float
    estimated_cost: float
    selected: bool


@dataclass(frozen=True)
class SubstrateTransition:
    step: int
    state: CanonicalState
    active_regions: tuple[str, ...]
    candidates: tuple[ComputationCandidate, ...]
    allocations: tuple[ResourceAllocation, ...]
    residency: ResidencySnapshot
    reason: str
    timestamp: float = field(default_factory=monotonic)


class NeuralSubstrateCoordinator:
    """Closed-loop coordinator across cognition, routing, compute and memory.

    The coordinator is intentionally not an authority engine. It can rank and
    allocate resources, but authoritative actions still have to pass through
    NSA's existing hard-state/capability machinery.
    """

    def __init__(
        self,
        state: Optional[CanonicalState] = None,
        residency: Optional[NeuralResidencyManager] = None,
        governor: Optional[GovernedSubstrateGovernor] = None,
        precision_by_tier: Optional[Mapping[MemoryTier, str]] = None,
    ) -> None:
        self.governor = governor or GovernedSubstrateGovernor(state or CanonicalState())
        self.residency = residency or NeuralResidencyManager(
            policy=ResidencyPolicy(
                vram_budget_bytes=1024**3,
                ram_budget_bytes=4 * 1024**3,
            )
        )
        self.precision_by_tier = dict(precision_by_tier or {
            MemoryTier.VRAM: "fp16",
            MemoryTier.RAM: "int8",
            MemoryTier.NVME: "int4",
        })
        self.history: list[SubstrateTransition] = []
        self._last_routing: dict[str, float] = {}
        self._tokens: list[TokenEnvelope] = []

    @property
    def state(self) -> CanonicalState:
        return self.governor.state

    def observe_token(self, token: TokenEnvelope) -> None:
        """Record token-flow context without changing authority."""
        self._tokens.append(token)
        if len(self._tokens) > 1024:
            self._tokens.pop(0)

    def evaluate_counterfactuals(self, simulator: Any, candidates: Sequence[Any]) -> Any:
        """Evaluate candidate actions before resource allocation or execution."""
        return simulator.evaluate(self.state, candidates)

    def consolidate_memory(self, consolidator: Any, trajectory: Any) -> Any:
        """Run episodic-to-semantic consolidation through the existing memory subsystem."""
        return consolidator.consolidate(trajectory, self.state, residency_manager=self.residency)

    def telemetry(self) -> Mapping[str, Any]:
        snapshot = self.residency.snapshot()
        return {
            "state_step": self.state.step,
            "tokens_observed": len(self._tokens),
            "active_region": self.residency.current_region,
            "routing_predictions": dict(self._last_routing),
            "residency": snapshot.to_dict(),
            "history_steps": len(self.history),
        }

    def observe_routing(
        self,
        regions: Sequence[str],
        probabilities: Sequence[float],
    ) -> None:
        """Record router evidence and feed it into the residency predictor."""
        self._last_routing = {
            rid: max(0.0, min(1.0, float(prob)))
            for rid, prob in zip(regions, probabilities)
        }
        observer = getattr(self.residency.predictor, "observe_routing", None)
        if observer is not None:
            observer(tuple(regions), tuple(probabilities))

    def plan(
        self,
        cognitive_state: Optional[Mapping[str, object]] = None,
        *,
        active_regions: Sequence[str] = (),
        utility_by_region: Optional[Mapping[str, float]] = None,
        latency_by_region: Optional[Mapping[str, float]] = None,
        reason: str = "substrate-plan",
    ) -> SubstrateTransition:
        """Plan compute and physical residency under the current state."""
        features = cognitive_state or self.state.summary()
        decisions = self.residency.plan(features)
        utilities = utility_by_region or {}
        latencies = latency_by_region or {}

        candidates = []
        for decision in decisions:
            probability = self.residency.scores.get(decision.region_id, 0.0)
            candidates.append(
                ComputationCandidate(
                    region_id=decision.region_id,
                    probability=probability,
                    utility=float(utilities.get(decision.region_id, 1.0)),
                    latency_ms=float(latencies.get(decision.region_id, 0.0)),
                    bytes_required=self.residency.regions[decision.region_id].size_bytes,
                    reason=decision.reason,
                )
            )

        allocations = self._allocate(decisions, candidates)
        snapshot = self.residency.snapshot()
        transition = SubstrateTransition(
            step=self.state.step,
            state=self.state,
            active_regions=tuple(active_regions),
            candidates=tuple(candidates),
            allocations=tuple(allocations),
            residency=snapshot,
            reason=reason,
        )
        self.history.append(transition)
        return transition

    def _allocate(
        self,
        decisions: Sequence[ResidencyDecision],
        candidates: Sequence[ComputationCandidate],
    ) -> list[ResourceAllocation]:
        by_id = {candidate.region_id: candidate for candidate in candidates}
        allocations: list[ResourceAllocation] = []
        vram_left = self.residency.policy.vram_budget_bytes
        ram_left = self.residency.policy.ram_budget_bytes

        for decision in sorted(decisions, key=lambda item: item.score, reverse=True):
            candidate = by_id[decision.region_id]
            tier = decision.desired_tier
            if tier == MemoryTier.VRAM and candidate.bytes_required > vram_left:
                tier = MemoryTier.RAM
            if tier == MemoryTier.RAM and candidate.bytes_required > ram_left:
                tier = MemoryTier.NVME

            if tier == MemoryTier.VRAM:
                vram_left -= candidate.bytes_required
            elif tier == MemoryTier.RAM:
                ram_left -= candidate.bytes_required

            benefit = candidate.probability * max(0.0, candidate.utility)
            cost = (
                candidate.bytes_required / max(1, self.residency.policy.vram_budget_bytes)
                + candidate.latency_ms / 1000.0
            )
            allocations.append(
                ResourceAllocation(
                    region_id=candidate.region_id,
                    tier=tier,
                    precision=self.precision_by_tier[tier],
                    probability=candidate.probability,
                    expected_utility=benefit,
                    estimated_cost=cost,
                    selected=benefit > cost,
                )
            )
        return allocations

    def commit_execution(
        self,
        active_regions: Sequence[str],
        *,
        reason: str = "computation-complete",
    ) -> SubstrateState:
        """Commit an observed computation/memory step through the governance boundary."""
        state = self.governor.commit_step(
            active_regions=active_regions,
            residency_snapshot=self.residency.snapshot(),
            reason=reason,
        )
        self._learn_execution(active_regions)
        return state

    def _learn_execution(self, active_regions: Sequence[str]) -> None:
        previous = self.residency.current_region
        observer = getattr(self.residency.predictor, "observe_transition", None)
        for region_id in active_regions:
            if observer is not None:
                observer(previous, region_id)
            previous = region_id
            self.residency.current_region = region_id

    def apply_resource_allocations(self, allocations: Sequence[ResourceAllocation]) -> None:
        """Apply only physical residency decisions; never changes hard authority."""
        for allocation in allocations:
            if not allocation.selected:
                continue
            if allocation.tier == MemoryTier.NVME:
                self.residency.record_evicted(
                    allocation.region_id, reason="coordinator-cold-allocation"
                )
            else:
                self.residency.record_resident(
                    allocation.region_id,
                    allocation.tier,
                    reason="coordinator-allocation",
                )

    def resource_pressure(self) -> float:
        snapshot = self.residency.snapshot()
        capacity = self.residency.policy.vram_budget_bytes + self.residency.policy.ram_budget_bytes
        used = (
            snapshot.bytes_by_tier.get(MemoryTier.VRAM, 0)
            + snapshot.bytes_by_tier.get(MemoryTier.RAM, 0)
        )
        return 0.0 if capacity <= 0 else min(1.0, used / capacity)
