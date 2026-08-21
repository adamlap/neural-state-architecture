"""NSA-backed governance for CCE.

CCE consumes NSA's public algebra/policy primitives without modifying nsa/.
Capabilities and thresholds are deployment data, never a hard-coded action list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from nsa.algebra import ConfidentialityLabel, IntegrityLabel, ProductLattice, ProductStateVector

from .action import ActionProposal, GovernanceDecision
from .state import CCEState


@dataclass(frozen=True)
class CCEPolicy:
    capabilities: frozenset[str] = field(default_factory=frozenset)
    minimum_confidence: float = 0.0
    maximum_risk: float = 1.0
    require_reversible: bool = False
    require_human_approval: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_environment(cls) -> "CCEPolicy":
        capabilities = frozenset(x.strip() for x in os.getenv("CCE_CAPABILITIES", "").split(",") if x.strip())
        approval = frozenset(x.strip() for x in os.getenv("CCE_HUMAN_APPROVAL", "").split(",") if x.strip())
        return cls(
            capabilities=capabilities,
            minimum_confidence=float(os.getenv("CCE_MIN_CONFIDENCE", "0.0")),
            maximum_risk=float(os.getenv("CCE_MAX_RISK", "1.0")),
            require_reversible=os.getenv("CCE_REQUIRE_REVERSIBLE", "0").lower() in {"1", "true", "yes"},
            require_human_approval=approval,
        )

    def allows_capability(self, capability: str) -> bool:
        return capability in self.capabilities


class CCEGovernor:
    def __init__(self, policy: CCEPolicy, lattice: ProductLattice | None = None):
        self.policy = policy
        self.lattice = lattice or ProductLattice()

    @classmethod
    def from_environment(cls) -> "CCEGovernor":
        return cls(CCEPolicy.from_environment())

    def source_state(self, state: CCEState) -> ProductStateVector:
        return ProductStateVector(confidentiality=ConfidentialityLabel.PUBLIC, integrity=IntegrityLabel.TRUSTED, authorizations=frozenset(self.policy.capabilities), confidence=max(0.0, min(1.0, state.self_confidence)), provenance=frozenset({"cce", "runtime"}), license_tier=0)

    def target_state(self, proposal: ActionProposal) -> ProductStateVector:
        return ProductStateVector(confidentiality=ConfidentialityLabel.PUBLIC, integrity=IntegrityLabel.TRUSTED, authorizations=frozenset({proposal.capability}), confidence=max(0.0, min(1.0, proposal.confidence)), provenance=frozenset({proposal.provenance}), license_tier=0)

    def evaluate(self, proposal: ActionProposal, source_state: ProductStateVector, target_state: ProductStateVector, *, human_approved: bool = False) -> GovernanceDecision:
        if not self.policy.allows_capability(proposal.capability):
            return GovernanceDecision("DENY", "capability not granted by deployment policy", proposal)
        if proposal.confidence < self.policy.minimum_confidence:
            return GovernanceDecision("HOLD", "confidence below policy threshold", proposal)
        if proposal.risk > self.policy.maximum_risk:
            return GovernanceDecision("DENY", "risk exceeds policy threshold", proposal)
        if self.policy.require_reversible and not proposal.reversible:
            return GovernanceDecision("DENY", "irreversible proposal forbidden", proposal)
        if proposal.capability in self.policy.require_human_approval and not human_approved:
            return GovernanceDecision("HOLD", "human approval required", proposal)
        if proposal.capability not in source_state.authorizations:
            return GovernanceDecision("DENY", "authorization absent from NSA source state", proposal)
        if not self.lattice.is_allowed(source_state, target_state):
            return GovernanceDecision("DENY", "NSA product-lattice transition forbidden", proposal)
        return GovernanceDecision("ALLOW", "NSA product-lattice transition permitted", proposal)
