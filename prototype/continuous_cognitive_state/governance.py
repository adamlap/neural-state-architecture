"""Runtime action governor backed by configurable NSA-compatible policy.

No action names are privileged here. A deployment supplies capability grants
and thresholds; the model can request a capability but cannot grant itself one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .action import ActionProposal, GovernanceDecision


@dataclass(frozen=True)
class CapabilityPolicy:
    granted: FrozenSet[str] = frozenset()
    max_risk: float = 0.0
    min_confidence: float = 1.0
    require_reversible: bool = False
    require_human_approval: FrozenSet[str] = frozenset()


@dataclass
class NSAGovernor:
    policy: CapabilityPolicy

    def evaluate(self, proposal: ActionProposal, *, human_approved: bool = False) -> GovernanceDecision:
        if proposal.capability not in self.policy.granted:
            return GovernanceDecision("DENY", "capability_not_granted", proposal)
        if proposal.risk > self.policy.max_risk:
            return GovernanceDecision("HOLD", "risk_exceeds_policy", proposal)
        if proposal.confidence < self.policy.min_confidence:
            return GovernanceDecision("HOLD", "confidence_below_policy", proposal)
        if self.policy.require_reversible and not proposal.reversible:
            return GovernanceDecision("HOLD", "irreversible_action_requires_policy", proposal)
        if proposal.capability in self.policy.require_human_approval and not human_approved:
            return GovernanceDecision("HOLD", "human_approval_required", proposal)
        return GovernanceDecision("ALLOW", "policy_satisfied", proposal)
