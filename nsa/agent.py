"""Stable application-facing NSA runtime with optional cognitive substrate."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from nsa.cce.engine import CCEStatus, ContinuousCognitiveEngine
from nsa.cce.lifecycle import CognitiveInputEvent, StateCheckpointStore
from nsa.cognition.embodied import ActiveCognition, ActiveCognitionState
from nsa.cognition.substrate import CognitiveState, CognitiveSubstrate, CognitiveSwitches
from nsa.core.state import CanonicalState, GoalState, SemanticState
from nsa.enforcement import EvaluationContext, PolicyEngine
from nsa.policy import NSAPolicy

class ModelBackend(Protocol):
    model: str
    def generate(self, prompt: str, *, state: Mapping[str, Any] | None = None) -> str: ...

@dataclass(frozen=True)
class AgentResult:
    text: str; state: CanonicalState; decision: Any = None; trace_id: int = 0; blocked: bool = False

@dataclass(frozen=True)
class RuntimeConfig:
    history_limit: int = 12
    include_state_in_prompt: bool = True
    auto_update_semantic_state: bool = True
    auto_checkpoint: bool = False
    continuous_interval_seconds: float = 0.1
    continuous_enabled: bool = False
    continuous_fail_closed: bool = True
    cognitive_enabled: bool = False
    cognitive_workspace_capacity: int = 4
    cognitive_switches: CognitiveSwitches = CognitiveSwitches()
    embodied_enabled: bool = False
    def __post_init__(self) -> None:
        if self.history_limit < 0 or self.continuous_interval_seconds <= 0 or self.cognitive_workspace_capacity < 1:
            raise ValueError("invalid runtime configuration")

class NSARuntime:
    """Single public runtime; cognitive and active layers are state transitions, not runtimes."""
    def __init__(self, backend: ModelBackend, *, state: Optional[CanonicalState] = None, initial_state: Optional[Mapping[str, Any]] = None, policy: Optional[NSAPolicy] = None, policy_engine: Optional[PolicyEngine] = None, checkpoint: Optional[StateCheckpointStore] = None, config: Optional[RuntimeConfig] = None, continuous_transition: Optional[Callable[[CanonicalState], CanonicalState]] = None) -> None:
        self.backend = backend; self.config = config or RuntimeConfig(); self.state = state or self._state_from_mapping(initial_state or {})
        self.policy_engine = policy_engine or (PolicyEngine(policy) if policy else None); self.checkpoint = checkpoint; self.history: list[CognitiveInputEvent] = []; self.trace: list[dict[str, Any]] = []
        self.cognitive = CognitiveSubstrate(switches=self.config.cognitive_switches, workspace_capacity=self.config.cognitive_workspace_capacity) if self.config.cognitive_enabled else None
        self.cognitive_state = CognitiveState(switches=self.config.cognitive_switches); self.active = ActiveCognition() if self.config.embodied_enabled else None; self.active_state = ActiveCognitionState()
        self._continuous_transition = continuous_transition or self._continuous_state_maintenance
        self._cce = ContinuousCognitiveEngine(self.state, self._continuous_transition, interval_seconds=self.config.continuous_interval_seconds, enabled=self.config.continuous_enabled, fail_closed=self.config.continuous_fail_closed)
    @staticmethod
    def _state_from_mapping(values: Mapping[str, Any]) -> CanonicalState:
        semantic = SemanticState(dict(values)); goal = values.get("goal")
        if goal: return CanonicalState(semantic=semantic, goals=GoalState(goals=(str(goal),), active_goal=str(goal)))
        return CanonicalState(semantic=semantic)
    def _run_cognitive_transition(self, observation: Any, *, confidence: float = 1.0, action: Any = None) -> None:
        if self.cognitive is not None: self.cognitive_state = self.cognitive.transition(self.cognitive_state, observation, confidence=confidence, action=action)
        if self.active is not None:
            u = self.cognitive_state.prediction.uncertainty if self.cognitive is not None else 1.0 - confidence
            self.active_state = self.active.transition(self.active_state, uncertainty=u, candidate_actions=("observe", "reflect", "respond"), information_gain={"observe": u, "reflect": 0.5*u, "respond": 0.1}, expected_utility={"respond": 0.5, "reflect": 0.2, "observe": 0.1}, risk={"respond": 0.1}, observation=observation, chosen_action=action)
    def _continuous_state_maintenance(self, state: CanonicalState) -> CanonicalState:
        if self.cognitive is not None or self.active is not None: self._run_cognitive_transition(state.semantic.value, confidence=state.soft.confidence)
        return replace(state, provenance=state.provenance.extend(transformation="cce.heartbeat"), step=state.step + 1)
    def observe(self, payload: Any, *, source: str = "text", confidence: float = 1.0, provenance: str = "local") -> None:
        self.history.append(CognitiveInputEvent(payload, source=source, confidence=confidence, provenance=provenance)); self.state = replace(self.state, soft=replace(self.state.soft, confidence=confidence, uncertainty=1.0-confidence), provenance=self.state.provenance.extend(source=source), step=self.state.step+1); self._run_cognitive_transition(payload, confidence=confidence); self._cce.set_state(self.state)
    def _prompt(self, prompt: str) -> str:
        if not self.config.include_state_in_prompt: return prompt
        recent = self.history[-self.config.history_limit:] if self.config.history_limit else []
        context = "\n".join(f"- {e.source}: {e.payload}" for e in recent)
        return f"You are operating inside the Neural State Architecture runtime.\nCURRENT_STATE={self.state.summary()!r}\nCOGNITIVE_STATE={self.cognitive_state.to_dict() if self.cognitive else None!r}\nACTIVE_COGNITION={self.active_state.to_dict() if self.active else None!r}\nRECENT_OBSERVATIONS={context!r}\n\nUSER_INPUT={prompt}"
    def step(self, prompt: str, *, action: str = "generate", capabilities: Sequence[str] = (), protected_data: Sequence[str] = ()) -> AgentResult:
        self.observe(prompt); context = EvaluationContext(action=action, capabilities=frozenset(capabilities), protected_data=frozenset(protected_data), risk=self.state.soft.risk, uncertainty=self.state.soft.uncertainty); decision = None
        if self.policy_engine:
            decision = self.policy_engine.enforce(prompt, context=context, state=self.state)
            if decision.decision.value in {"deny", "require_approval"}: self.trace.append({"step": self.state.step, "prompt": prompt, "blocked": True}); return AgentResult("", self.state, decision=decision, trace_id=len(self.trace), blocked=True)
        text = self.backend.generate(self._prompt(prompt), state=self.state.summary())
        if self.config.auto_update_semantic_state:
            self.state = replace(self.state, semantic=SemanticState(text), provenance=self.state.provenance.extend(transformation="llm.generate"), step=self.state.step+1); self._run_cognitive_transition(text, confidence=self.state.soft.confidence, action=action)
        self._cce.set_state(self.state); self.trace.append({"step": self.state.step, "prompt": prompt, "response": text, "blocked": False});
        if self.checkpoint and self.config.auto_checkpoint: self.checkpoint.save(self.snapshot())
        return AgentResult(text, self.state, decision=decision, trace_id=len(self.trace))
    run = step
    def continuous_tick(self) -> bool:
        ok = self._cce.tick(); self.state = self._cce.state; return ok
    def continuous_start(self) -> bool: return self._cce.start()
    def continuous_stop(self, timeout: float | None = None) -> bool: return self._cce.stop(timeout=timeout)
    def continuous_set_enabled(self, enabled: bool) -> None: self._cce.set_enabled(enabled)
    def continuous_status(self) -> CCEStatus: return self._cce.status()
    @property
    def cce(self) -> ContinuousCognitiveEngine[CanonicalState]: return self._cce
    def cognitive_metrics(self) -> Mapping[str, float | int]:
        m=self.cognitive_state.metrics; r={"workspace_ignitions":m.workspace_ignitions,"broadcast_coverage":m.broadcast_coverage,"prediction_error":m.prediction_error,"integration":m.integration,"recurrence":m.recurrence,"self_model_accuracy":m.self_model_accuracy,"cognitive_continuity":m.cognitive_continuity,"information_gain":m.information_gain}
        if self.active: r.update({"active_information_gain":self.active_state.information_gain,"expected_free_energy":self.active_state.expected_free_energy,"identity_continuity":self.active_state.identity.continuity,"homeostatic_stability":self.active_state.homeostasis.stability})
        return r
    def select_action(self, actions: Sequence[str], *, information_gain: Mapping[str,float] | None=None, expected_utility: Mapping[str,float] | None=None, risk: Mapping[str,float] | None=None) -> str | None:
        if not self.active: raise RuntimeError("embodied cognition is disabled")
        self.active_state=self.active.transition(self.active_state, uncertainty=self.state.soft.uncertainty, candidate_actions=actions, information_gain=information_gain, expected_utility=expected_utility, risk=risk); return self.active_state.selected_action
    def snapshot(self) -> dict[str,Any]: return {"state":dict(self.state.summary()),"history":[e.payload for e in self.history],"trace":list(self.trace),"cognitive_state":self.cognitive_state.to_dict() if self.cognitive else None,"active_cognition":self.active_state.to_dict() if self.active else None}
    def save(self) -> None:
        if self.checkpoint is None: raise RuntimeError("no checkpoint store configured")
        self.checkpoint.save(self.snapshot())

NSA = NSARuntime
__all__=["AgentResult","ModelBackend","NSA","NSARuntime","RuntimeConfig"]
