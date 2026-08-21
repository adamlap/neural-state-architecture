"""NSA-backed governance for CCE.

No privileged capability names are embedded here. The deployment supplies the
capability set and thresholds. The governor uses NSA's ProductLattice and
HardStateVector as the authoritative state policy primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nsa.algebra import (
    ConfidentialityLabel,
    HardStateVector,
    IntegrityLabel,
    ProductLattice,
    ProductStateVector,
)

from ..action import ActionProposal, GovernanceDecision


@dataclass(frozen=True)
class CCEPolicy:
    """Deployment-supplied policy; it contains no fixed action catalogue."""

    capabilities: frozenset[str] = field(default_factory=frozenset)
    minimum_confidence: float = 0.0
    maximum_risk: float = 1.0
    require_reversible: bool = False
    require_human_approval: frozenset[str] = field(default_factory=frozenset)

    def allows_capability(self, capability: str) -> bool:
        return capability in self.capabilities


class CCEGovernor:
    """Evaluate live CCE proposals against NSA product-state constraints."""

    def __init__(self, policy: CCEPolicy, lattice: ProductLattice | None = None):
        self.policy = policy
        self.lattice = lattice or ProductLattice()

    def evaluate(
        self,
        proposal: ActionProposal,
        source_state: ProductStateVector,
        target_state: ProductStateVector,
        *,
        human_approved: bool = False,
    ) -> GovernanceDecision:
        if not self.policy.allows_capability(proposal.capability):
            return GovernanceDecision("DENY", "capability not granted by deployment policy", proposal)
        if proposal.confidence < self.policy.minimum_confidence:
            return GovernanceDecision("HOLD", "proposal confidence below policy threshold", proposal)
        if proposal.risk > self.policy.maximum_risk:
            return GovernanceDecision("DENY", "proposal risk exceeds policy threshold", proposal)
        if self.policy.require_reversible and not proposal.reversible:
            return GovernanceDecision("DENY", "irreversible proposal forbidden by policy", proposal)
        if proposal.capability in self.policy.require_human_approval and not human_approved:
            return GovernanceDecision("HOLD", "human approval required by deployment policy", proposal)
        if not self.lattice.is_allowed(source_state, target_state):
            return GovernanceDecision("DENY", "NSA product-lattice transition is forbidden", proposal)
        return GovernanceDecision("ALLOW", "NSA product-lattice transition and deployment policy allow proposal", proposal)

    @staticmethod
    def default_source_state() -> ProductStateVector:
        return ProductStateVector(
            confidentiality=ConfidentialityLabel.PUBLIC,
            integrity=IntegrityLabel.TRUSTED,
            authorizations=frozenset(),
            confidence=1.0,
            provenance=frozenset({"cce"}),
            license_tier=0,
        )

    @staticmethod
    def target_from_hard_state(hard: HardStateVector, confidence: float, provenance: frozenset[str]) -> ProductStateVector:
        return ProductStateVector(
            confidentiality=hard.confidentiality,
            integrity=hard.integrity,
            authorizations=hard.authorizations,
            confidence=max(0.0, min(1.0, confidence)),
            provenance=provenance,
            license_tier=hard.license_tier,
        )
