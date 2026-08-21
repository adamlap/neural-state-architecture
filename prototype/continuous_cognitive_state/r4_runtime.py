"""R4 live runtime boundary: dynamic cognition -> NSA-governed actions."""
from __future__ import annotations

from dataclasses import dataclass

from .action import GovernanceDecision
from .dynamics import DynamicalState
from .governance import NSAGovernor
from .ollama_actions import OllamaActionReasoner
from .state import CognitiveState


@dataclass
class R4Runtime:
    state: CognitiveState
    dynamics: DynamicalState
    reasoner: OllamaActionReasoner
    governor: NSAGovernor

    def tick(self, event: str | None = None) -> GovernanceDecision | None:
        drive = 0.0 if not event else min(1.0, len(event) / 256.0)
        self.dynamics.tick(drive=drive)
        self.state.tick_once()
        self.state.last_input = event
        self.state.cognitive_context = event or self.state.cognitive_context
        proposal = self.reasoner.propose(self.state, event)
        if proposal is None:
            return None
        decision = self.governor.evaluate(proposal)
        self.state.governance.authorized = decision.allowed
        self.state.governance.risk = proposal.risk
        self.state.governance.provenance = proposal.provenance
        return decision
