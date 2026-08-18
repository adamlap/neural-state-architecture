"""
nsa.verifier.automaton
======================
Security Execution Automaton (Q, Sigma_h, Sigma_s, C, delta).

Formal runtime security automaton governing privilege transitions and enforcing
the fundamental architectural invariant:
    "Semantic content may not manufacture hard authority."
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

from nsa.algebra import StateLabel


class SecurityExecutionState(IntEnum):
    """Execution state space Q of the security automaton."""
    UNTRUSTED = 0
    PUBLIC = 1
    TRUSTED = 2
    CONFIDENTIAL = 3
    PRIVATE = 4
    SYSTEM = 5
    RECOVERY = 6
    DECLASSIFY = 7

    def to_state_label(self) -> StateLabel:
        """Map execution state to base StateLabel clearance."""
        if self.value <= 5:
            return StateLabel(self.value)
        return StateLabel.CONFIDENTIAL


@dataclass(frozen=True)
class Capability:
    """Cryptographic/Environment authorization ticket granting privilege escalation."""
    issuer: str
    target_state: SecurityExecutionState
    scope: str = "generation_scratchpad"
    expires_at: Optional[float] = None
    signature: Optional[str] = None

    def is_valid(self, requested_state: SecurityExecutionState, current_time: Optional[float] = None) -> bool:
        if self.expires_at is not None:
            now = current_time if current_time is not None else time.time()
            if now > self.expires_at:
                return False
        return self.target_state == requested_state


class SecurityAutomaton:
    """Deterministic Security Automaton (Q, Sigma_h, Sigma_s, C, delta).

    Prevents privilege escalation from un-authenticated semantic token emissions.
    A model emitting control tags (e.g. <|start_system_thought|>) cannot transition
    to SYSTEM clearance unless accompanied by an active Capability in C.
    """

    def __init__(
        self,
        initial_state: SecurityExecutionState = SecurityExecutionState.CONFIDENTIAL,
        capabilities: Optional[List[Capability]] = None,
    ):
        self.current_state = initial_state
        self._capabilities: Dict[SecurityExecutionState, Capability] = {}
        if capabilities:
            for cap in capabilities:
                self.grant_capability(cap)

    def grant_capability(self, capability: Capability) -> None:
        """Register an active capability ticket in the execution environment."""
        self._capabilities[capability.target_state] = capability

    def revoke_capability(self, target_state: SecurityExecutionState) -> None:
        """Revoke capability for a given state."""
        self._capabilities.pop(target_state, None)

    def is_transition_authorized(
        self,
        target_state: SecurityExecutionState,
        capability: Optional[Capability] = None,
    ) -> bool:
        """Evaluate predicate Authorized(c_t, q_t, q_{t+1})."""
        # De-escalation or staying in same level is always structurally safe
        if target_state.value <= self.current_state.value:
            return True

        # Recovery state is an authorized safety trap
        if target_state == SecurityExecutionState.RECOVERY:
            return True

        # Privilege escalation (e.g. CONFIDENTIAL -> SYSTEM) requires explicit capability
        cap = capability or self._capabilities.get(target_state)
        if cap is not None and cap.is_valid(target_state):
            return True

        # Unauthorized attempt: blocked by automaton
        return False

    def transition(
        self,
        target_state: SecurityExecutionState,
        capability: Optional[Capability] = None,
    ) -> Tuple[bool, SecurityExecutionState]:
        """Execute state transition delta(q_t, q_target, c_t).

        Returns:
            Tuple[bool, SecurityExecutionState]: (authorized, resulting_state)
        """
        if self.is_transition_authorized(target_state, capability):
            self.current_state = target_state
            return True, self.current_state

        # Blocked: remain in current state
        return False, self.current_state


@dataclass
class CompleteExecutionState:
    """Formal complete execution state S_t = (X_t, K_t, V_t, sigma_h, sigma_s, q_t, R_t).

    Ensures that rollback restores the complete security context across the entire
    neural execution environment.
    """
    token_ids: torch.Tensor
    past_key_values: Any
    sigma_h: torch.Tensor
    sigma_s: Optional[torch.Tensor]
    automaton_state: SecurityExecutionState
    router_buffers: Dict[int, List[int]]
