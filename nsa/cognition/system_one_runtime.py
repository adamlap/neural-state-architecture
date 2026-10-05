"""Provider-neutral, continuous System 1 control plane for NSA."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Mapping, Protocol, Sequence

from nsa.cognition.system_one import SystemOneDecisionEngine, TypedDecisionSchema
from nsa.cognition.tools import ToolRegistry
from nsa.core.state import CanonicalState


@dataclass(frozen=True)
class DecisionQuestion:
    """A typed, generation-free System 1 decision request."""

    name: str
    question: str
    choices: tuple[str, ...]
    descriptions: Mapping[str, str] = field(default_factory=dict)
    min_confidence: float = 0.5
    risk_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.question.strip():
            raise ValueError("name and question must be non-empty")
        if not self.choices or len(set(self.choices)) != len(self.choices):
            raise ValueError("choices must be non-empty and unique")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class SystemOneDecision:
    """Provider-neutral result suitable for routing and control."""

    question: str
    selected_choice: str
    probabilities: Mapping[str, float]
    confidence: float
    uncertainty: float
    risk: float
    passed_gate: bool
    backend: str
    latency_ms: float
    reason: str = ""

    @property
    def should_escalate(self) -> bool:
        return not self.passed_gate or self.uncertainty >= 0.65


@dataclass(frozen=True)
class SystemOneTick:
    """One continuous cognition pass; no text generation is required."""

    decisions: Mapping[str, SystemOneDecision]
    state_soft_updates: Mapping[str, float]
    escalate: bool
    selected_tool: str | None = None
    selected_model: str | None = None
    memory_policy: str | None = None
    salience: float = 0.0


class SystemOneBackend(Protocol):
    """Backend contract implemented by deterministic, hosted, or local models."""

    name: str

    def decide(
        self,
        question: DecisionQuestion,
        state: CanonicalState,
        *,
        features: Mapping[str, float] | None = None,
    ) -> SystemOneDecision:
        ...


class DeterministicSystemOneBackend:
    """Fast built-in backend with no ML dependency."""

    name = "deterministic"

    def __init__(self, engine: SystemOneDecisionEngine | None = None) -> None:
        self.engine = engine or SystemOneDecisionEngine()

    def decide(self, question: DecisionQuestion, state: CanonicalState, *, features=None) -> SystemOneDecision:
        started = monotonic()
        signals = dict(features or {})
        if not signals:
            base = 1.0 - state.soft.uncertainty
            signals = {choice: base for choice in question.choices}
        schema = TypedDecisionSchema(
            name=question.name,
            choices=question.choices,
            descriptions=question.descriptions,
            min_confidence=question.min_confidence,
            risk_weights=question.risk_weights,
        )
        result = self.engine.evaluate_schema(schema, signals)
        return SystemOneDecision(
            question=question.name,
            selected_choice=result.selected_choice,
            probabilities=result.probabilities,
            confidence=result.calibrated_confidence,
            uncertainty=result.epistemic_uncertainty,
            risk=result.risk_score,
            passed_gate=result.passed_gate,
            backend=self.name,
            latency_ms=max(result.latency_ms, (monotonic() - started) * 1000.0),
            reason=result.reason,
        )


class CallableSystemOneBackend:
    """Adapter for Jev-like hosted services and custom local decision models.

    The callable returns choice probabilities. NSA normalizes and audits them;
    the callable never receives authority to mutate canonical state.
    """

    def __init__(self, decide_fn: Callable[..., Mapping[str, float]], *, name: str = "callable") -> None:
        self.decide_fn = decide_fn
        self.name = name

    def decide(self, question: DecisionQuestion, state: CanonicalState, *, features=None) -> SystemOneDecision:
        started = monotonic()
        raw = self.decide_fn(question=question, state=state, features=features or {})
        probabilities = {choice: max(0.0, float(raw.get(choice, 0.0))) for choice in question.choices}
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("System 1 backend returned no probability mass")
        probabilities = {k: v / total for k, v in probabilities.items()}
        selected = max(probabilities, key=probabilities.get)
        confidence = probabilities[selected]
        uncertainty = self._entropy(tuple(probabilities.values()))
        risk = float(question.risk_weights.get(selected, 0.0))
        passed = confidence >= question.min_confidence
        return SystemOneDecision(
            question=question.name,
            selected_choice=selected,
            probabilities=probabilities,
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
            passed_gate=passed,
            backend=self.name,
            latency_ms=(monotonic() - started) * 1000.0,
            reason="" if passed else f"confidence {confidence:.2f} < gate {question.min_confidence:.2f}",
        )

    @staticmethod
    def _entropy(probs: Sequence[float]) -> float:
        import math
        if len(probs) <= 1:
            return 0.0
        h = -sum(p * math.log(p) for p in probs if p > 1e-12)
        return max(0.0, min(1.0, h / math.log(len(probs))))


class SystemOneController:
    """Reusable System 1 control plane for CCE, agents, routing and tools."""

    def __init__(self, backend: SystemOneBackend | None = None) -> None:
        self.backend = backend or DeterministicSystemOneBackend()

    def decide(self, question: DecisionQuestion, state: CanonicalState, *, features=None) -> SystemOneDecision:
        return self.backend.decide(question, state, features=features)

    def choose_tool(self, state: CanonicalState, registry: ToolRegistry, *, objective: str = "choose the safest useful tool") -> SystemOneDecision:
        tools = [tool for tool in registry.all() if state.hard.has_permission(tool.capability)]
        if not tools:
            return self.decide(DecisionQuestion("tool_route", objective, ("none",), min_confidence=0.0), state, features={"none": 1.0})
        choices = tuple(tool.name for tool in tools) + ("none",)
        descriptions = {tool.name: f"{tool.capability}; risk={tool.risk:.2f}; reversible={tool.reversible}" for tool in tools}
        descriptions["none"] = "do not invoke a tool"
        question = DecisionQuestion("tool_route", objective, choices, descriptions, 0.55, {tool.name: tool.risk for tool in tools})
        features = {tool.name: 1.0 - tool.risk for tool in tools}
        features["none"] = max(0.05, state.soft.uncertainty * 0.25)
        return self.decide(question, state, features=features)

    def choose_model(self, state: CanonicalState, models: Sequence[str], *, objective: str = "choose the least expensive capable model", scores=None) -> SystemOneDecision:
        choices = tuple(models)
        if not choices:
            raise ValueError("models must not be empty")
        return self.decide(DecisionQuestion("model_route", objective, choices, min_confidence=0.55), state, features=scores or {m: 1.0 for m in choices})

    def choose_action(self, state: CanonicalState, candidates: Sequence[Any]) -> SystemOneDecision:
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("candidates must not be empty")
        choices = tuple(candidate.action_id for candidate in candidates)
        question = DecisionQuestion(
            "action_route", "choose the best executable action", choices, min_confidence=0.50,
            risk_weights={candidate.action_id: candidate.risk for candidate in candidates},
        )
        goals = tuple(g.lower() for g in state.goals.goals)
        features = {
            candidate.action_id: max(
                0.0, min(1.0, candidate.expected_utility - candidate.risk +
                (0.15 if candidate.action_id.lower() in goals else 0.0))
            )
            for candidate in candidates
        }
        return self.decide(question, state, features=features)

    def tick(
        self,
        state: CanonicalState,
        *,
        observations: Sequence[Any] = (),
        tools: ToolRegistry | None = None,
        models: Sequence[str] = (),
        model_scores: Mapping[str, float] | None = None,
    ) -> SystemOneTick:
        """Run a generation-free continuous cognition heartbeat."""
        decisions: dict[str, SystemOneDecision] = {}
        if tools is not None:
            decisions["tool_route"] = self.choose_tool(state, tools)
        if models:
            decisions["model_route"] = self.choose_model(state, models, scores=model_scores)

        salience = min(1.0, state.soft.uncertainty * 0.6 + state.soft.risk * 0.4)
        if observations:
            salience = min(1.0, salience + 0.15)

        decisions["escalation"] = self.decide(
            DecisionQuestion(
                "escalation", "is expensive System 2 reasoning required now?",
                ("escalate", "continue_system_one"), min_confidence=0.60,
                risk_weights={"continue_system_one": state.soft.risk},
            ),
            state,
            features={
                "escalate": min(1.0, state.soft.uncertainty * 1.2 + state.soft.risk * 0.4 + salience * 0.3),
                "continue_system_one": max(0.0, 1.0 - state.soft.uncertainty),
            },
        )
        decisions["memory_policy"] = self.decide(
            DecisionQuestion(
                "memory_policy", "should this state be retained beyond the current working cycle?",
                ("discard", "working", "episodic", "durable"), min_confidence=0.55,
            ),
            state,
            features={
                "discard": max(0.0, 1.0 - salience),
                "working": 0.4 + salience,
                "episodic": salience * 0.9 + state.soft.uncertainty * 0.2,
                "durable": max(0.0, salience + state.goals.priority - 0.5),
            },
        )
        decisions["salience"] = self.decide(
            DecisionQuestion("salience", "how much attention should the current state receive?", ("low", "normal", "high"), min_confidence=0.0),
            state,
            features={"low": max(0.0, 1.0 - salience), "normal": 0.5, "high": salience + 0.25},
        )

        escalation = decisions["escalation"].selected_choice == "escalate"
        updates = {
            "uncertainty": max(state.soft.uncertainty, decisions["escalation"].uncertainty),
            "risk": max(state.soft.risk, decisions["escalation"].risk),
            "confidence": min(state.soft.confidence, decisions["escalation"].confidence),
        }
        return SystemOneTick(
            decisions=decisions,
            state_soft_updates=updates,
            escalate=escalation,
            selected_tool=decisions.get("tool_route").selected_choice if "tool_route" in decisions else None,
            selected_model=decisions.get("model_route").selected_choice if "model_route" in decisions else None,
            memory_policy=decisions["memory_policy"].selected_choice,
            salience=salience,
        )


__all__ = ["CallableSystemOneBackend", "DecisionQuestion", "DeterministicSystemOneBackend",
           "SystemOneBackend", "SystemOneController", "SystemOneDecision", "SystemOneTick"]
