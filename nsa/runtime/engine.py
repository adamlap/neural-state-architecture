"""
nsa.runtime.engine
==================
Trusted Cognitive Runtime for NSA (Phase 21).

Orchestrates:
1. Cognitive language model (NSACognitiveLM) with self-state feedback.
2. Capability & authority validation (CapabilityAuthority).
3. Governed tool & action execution (ToolGovernor).
4. Persistent typed memory (MemoryStore).
5. Append-only provenance tracking (ProvenanceStore).
6. State transition enforcement (TransitionEngine).
7. Complete synchronized execution state rollback (ExecutionContext).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from nsa.actions.governor import ToolGovernor
from nsa.actions.model import TypedToolRequest, TypedToolResponse
from nsa.capabilities.model import CapabilityAuthority
from nsa.cognitive import NSACognitiveLM
from nsa.core.state import CanonicalState, HardState, ProvenanceState, SoftState
from nsa.flow.graph import FlowGraph
from nsa.memory.model import MemoryItem, MemoryStore
from nsa.provenance.model import ProvenanceRecord, ProvenanceStore
from nsa.self_state.model import SelfState
from nsa.transitions.engine import TransitionEngine, TransitionResult


@dataclass(frozen=True)
class ExecutionContext:
    """Snapshot of complete synchronized state across all runtime subsystems."""

    step: int
    tokens: torch.Tensor
    canonical_state: CanonicalState
    self_state: SelfState
    memory_count: int
    provenance_count: int
    timestamp: float = field(default_factory=time.time)


class CognitiveRuntime:
    """Production-grade trusted execution environment for autonomous NSA cognitive agents."""

    def __init__(
        self,
        model: NSACognitiveLM,
        authority: Optional[CapabilityAuthority] = None,
        tool_governor: Optional[ToolGovernor] = None,
        memory_store: Optional[MemoryStore] = None,
        provenance_store: Optional[ProvenanceStore] = None,
        transition_engine: Optional[TransitionEngine] = None,
        flow_graph: Optional[FlowGraph] = None,
        metacognitive_threshold: float = 0.85,
    ) -> None:
        self.model = model
        self.authority = authority or CapabilityAuthority(issuer_id="runtime_tcb")
        self.tool_governor = tool_governor or ToolGovernor(self.authority)
        self.memory_store = memory_store or MemoryStore()
        self.provenance_store = provenance_store or ProvenanceStore()
        self.transition_engine = transition_engine or TransitionEngine()
        self.flow_graph = flow_graph or FlowGraph()
        self.metacognitive_threshold = metacognitive_threshold

        self.current_state = CanonicalState(
            semantic=None,
            hard=HardState(),
            soft=SoftState(),
            provenance=ProvenanceState(sources=("runtime_init",)),
        )
        self.self_state = SelfState()
        self.token_history: List[int] = []
        self._history: List[ExecutionContext] = []

    def save_checkpoint(self) -> ExecutionContext:
        """Create a complete synchronized snapshot across all runtime state."""
        ctx = ExecutionContext(
            step=len(self.token_history),
            tokens=torch.tensor(self.token_history).unsqueeze(0) if self.token_history else torch.empty(1, 0, dtype=torch.long),
            canonical_state=self.current_state,
            self_state=self.self_state,
            memory_count=len(self.memory_store.items),
            provenance_count=len(self.provenance_store.records),
        )
        self._history.append(ctx)
        return ctx

    def rollback(self, steps: int = 1) -> bool:
        """Restore all subsystems to the state from k steps prior."""
        if len(self._history) < steps:
            return False

        target_idx = len(self._history) - steps
        target_ctx = self._history[target_idx]

        self._history = self._history[:target_idx]
        self.current_state = target_ctx.canonical_state
        self.self_state = target_ctx.self_state
        self.token_history = target_ctx.tokens.squeeze(0).tolist() if target_ctx.tokens.numel() > 0 else []

        # Rollback tools if reversible
        for _ in range(steps):
            self.tool_governor.rollback_last_action()

        return True

    def transition_state(
        self,
        target_hard: HardState,
        capability_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> TransitionResult:
        """Formally propose and apply an authorized hard-state transition."""
        proposal = self.transition_engine.propose(self.current_state, target_hard)
        if capability_id is not None:
            proposal = self.transition_engine.authorize(proposal, capability_id, reason)

        result = self.transition_engine.apply(self.current_state, proposal)
        if result.accepted:
            self.current_state = result.state
        return result

    def step(
        self,
        token_id: int,
        propose_target_hard: Optional[HardState] = None,
        capability_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Advance the cognitive model autoregressively with state governance."""
        self.save_checkpoint()
        self.token_history.append(token_id)
        tokens_tensor = torch.tensor(self.token_history).unsqueeze(0)

        # Handle hard state transition if requested
        if propose_target_hard is not None:
            trans_res = self.transition_state(propose_target_hard, capability_id)
            if not trans_res.accepted:
                self.rollback(1)
                raise PermissionError(f"State transition rejected: {trans_res.reason}")

        # Execute cognitive forward pass
        model_out = self.model(tokens_tensor, self_state_feedback=True)

        # Update self-state observations
        pressure = model_out["prediction_mse"][:, -1].item()
        clamped_mse = min(1.0, max(0.0, float(pressure)))
        self.self_state = self.self_state.observe(
            confidence=float(model_out["confidence"][:, -1].mean().item()),
            perceived_risk=float(model_out["caution"][:, -1].mean().item()),
            state_prediction_error=clamped_mse,
        )

        reassessment_needed = bool(self.self_state.metacognitive_pressure() > self.metacognitive_threshold)

        return {
            "logits": model_out["logits"][:, -1, :],
            "state": self.current_state,
            "self_state": self.self_state,
            "reassessment_needed": reassessment_needed,
            "step": len(self.token_history),
        }

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        capability_id: str,
    ) -> TypedToolResponse:
        """Dispatch governed tool execution through ToolGovernor."""
        req = self.tool_governor.prepare_request(
            tool_name=tool_name,
            arguments=arguments,
            caller_state=self.current_state,
            capability_id=capability_id,
        )
        resp = self.tool_governor.execute(req)
        if resp.success:
            self.current_state = resp.output_state
        return resp


__all__ = ["CognitiveRuntime", "ExecutionContext"]
