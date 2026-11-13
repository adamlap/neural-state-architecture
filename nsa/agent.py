"""Stable application-facing NSA runtime with optional cognitive substrate."""
from __future__ import annotations
import inspect
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from nsa.cce.engine import CCEStatus, ContinuousCognitiveEngine
from nsa.cce.canonical_runtime import CanonicalCCERuntime, TickInput
from nsa.cce.lifecycle import CognitiveInputEvent, StateCheckpointStore
from nsa.cognition.embodied import ActiveCognition, ActiveCognitionState
from nsa.cognition.substrate import CognitiveState, CognitiveSubstrate, CognitiveSwitches
from nsa.core.state import CanonicalState, GoalState, SemanticState
from nsa.algebra import IntegrityLabel
from nsa.enforcement import EvaluationContext, PolicyEngine
from nsa.policy import NSAPolicy
from nsa.core.state_codec import decode_state, encode_state
from nsa.cognition.system_one_runtime import SystemOneController, SystemOneBackend, SystemOneTick
from nsa.cognition.tools import ToolRegistry
from nsa.memory import TemporalMemoryStore
from nsa.memory.extract import FactExtractor, rank_keys, rank_scored
from nsa.memory.routing import route_fact
from nsa.decision import SecurityDecision
from nsa.taint import authorize_tool_call, join_integrity
from nsa.ingress import IngressAuthority, IngressEnvelope, content_digest

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
    # Whether recent observations are shown to the model. None follows
    # include_state_in_prompt (the historical behaviour). Set True with
    # include_state_in_prompt=False for a matched "history only" control that
    # sees the same window as the state-bearing prompt but no state text.
    include_history_in_prompt: Optional[bool] = None
    # Whether recent observations are still shown when retrieved memory is attached to
    # the prompt. True keeps the historical behaviour. In a keyed-recall benchmark the
    # unrelated recent observations compete with the retrieved value (qwen2.5:1.5b:
    # 5/18 correct with them, 18/18 without), so tasks that answer from memory can turn
    # it off; conversational use usually wants them.
    include_history_with_memory: bool = True
    # Fold provenance older than the last N entries into a constant-size hash chain (see ProvenanceState.fold). None keeps the full history in state; set it (e.g. 256)
    # for long-lived agents: without it every observe() re-hashes the whole provenance, so per-call cost grows linearly with agent age.
    provenance_window: Optional[int] = None
    # Observation sources whose content is labelled UNTRUSTED at ingress (e.g. "web", "email",
    # "tool_output"). Empty keeps every observation trusted (historical behaviour).
    untrusted_sources: tuple[str, ...] = ()
    # With an IngressAuthority configured and this True, a TRUSTED label is accepted only from a
    # verified envelope: unsigned observe()/remember() calls are labelled UNTRUSTED, even if they
    # pass integrity=TRUSTED explicitly. See nsa/ingress.py.
    require_signed_ingress: bool = False
    # With a NoveltyGate configured: only lines whose novelty is at least this value reach the fact extractor (selective storage). None means
    # the gate only measures (``last_novelty``); every line is still extracted. Lines are scored one by one; the gate learns every line.
    novelty_write_threshold: Optional[float] = None
    # Route each extracted fact against the memory's current value for its key (nsa/memory/routing.py): ignore restatements, hold a near-duplicate
    # value without an update cue (likely noise) until it is observed twice, write everything else.
    route_noisy_updates: bool = False
    # Model-call gating in step() (after policy enforcement, so security still comes first). Only applies when memory is used for the step
    # (``memory_keys`` given or ``auto_memory=True``).
    #   abstain_text: if ``auto_memory`` finds no matching memory key, return this text WITHOUT calling the model.
    #   reuse_familiar_answers: if the retrieved memory state (keys and their versions) is unchanged since an earlier call and the question's wording is familiar
    #   (character-trigram cosine >= ``reuse_similarity``), return the earlier answer WITHOUT calling the model. Any write to a retrieved key invalidates it.
    # Both need NumPy only for reuse. Default off.
    abstain_text: Optional[str] = None
    reuse_familiar_answers: bool = False
    reuse_similarity: float = 0.95
    reuse_cache_size: int = 256
    # When True, memory_keys_for accepts only keys whose every token is covered by the question (typos tolerated). Stops "the database port for zebra" from
    # answering with the only entity that happens to have a database port. Recommended with abstain_text.
    require_full_key_match: bool = False
    # Cross-check memory against the raw text: with ``auto_memory`` (or memory_keys), also put the ``retrieve_lines`` most similar earlier observations into the prompt
    # as RETRIEVED_LINES. Extraction misses many facts on unfamiliar wording (recall 0.2-0.35); the lines cover what it missed. Measured (qwen2.5:1.5b, noisy streams):
    # memory + lines 0.92 / 0.79 (known / held-out phrasings) vs routed memory alone 0.79 / 0.56. 0 disables.
    retrieve_lines: int = 0
    # Speed (both off by default). ``compact_memory`` renders ``- key: value`` rows without VERSION/STATUS (fewer prompt tokens). ``pin_memory_prefix`` puts the WHOLE
    # current memory (up to ``pin_memory_max`` keys, first-seen order, compact) at the start of every memory-backed prompt instead of the retrieved rows: the text is the
    # same from call to call, so a server with a KV prefix cache (Ollama, llama.cpp) only evaluates the question. Measured 1.6x (7B, accuracy unchanged) and 1.8x (1.5B, but
    # accuracy 1.00 -> 0.79 with 80 distractor rows): use with a model that can pick a row out of the whole memory; a memory update in the middle re-evaluates the rest once.
    # ``auto_tune``: pick the measured-good options for the attached model (nsa/autotune.py): threads, provenance window, answer reuse, and by model size either a pinned
    # memory prefix (>= 7B) or raw-line fallback (< 7B). Explicitly set fields are never overridden.
    auto_tune: bool = False
    # ``answer_from_memory``: a plain lookup whose words are all covered by exactly ONE stored key (plus filler such as "what is the") is answered with the stored value and
    # no model call (after policy enforcement, like the other gates). Anything else in the question ("why", "previous", a second entity, a format request) goes to the model.
    answer_from_memory: bool = False
    compact_memory: bool = False
    pin_memory_prefix: bool = False
    pin_memory_max: int = 200
    auto_update_semantic_state: bool = True
    auto_checkpoint: bool = False
    continuous_interval_seconds: float = 0.1
    continuous_enabled: bool = False
    continuous_fail_closed: bool = True
    cognitive_enabled: bool = False
    cognitive_workspace_capacity: int = 4
    cognitive_switches: CognitiveSwitches = CognitiveSwitches()
    embodied_enabled: bool = False
    system_one_enabled: bool = False
    system_one_models: tuple[str, ...] = ()
    memory_enabled: bool = False
    memory_limit: int = 3
    def __post_init__(self) -> None:
        if self.history_limit < 0 or self.continuous_interval_seconds <= 0 or self.cognitive_workspace_capacity < 1 or self.memory_limit < 1: raise ValueError("invalid runtime configuration")

class NSARuntime:
    """Single public runtime; legacy cognition remains available while canonical CCE is exposed as the governed transaction plane."""
    def __init__(self, backend: ModelBackend, *, state: Optional[CanonicalState] = None, initial_state: Optional[Mapping[str, Any]] = None,
                 policy: Optional[NSAPolicy] = None, policy_engine: Optional[PolicyEngine] = None, checkpoint: Optional[StateCheckpointStore] = None,
                 config: Optional[RuntimeConfig] = None, continuous_transition: Optional[Callable[[CanonicalState], CanonicalState]] = None,
                 system_one: Optional[SystemOneController] = None, system_one_backend: Optional[SystemOneBackend] = None,
                 system_one_tools: Optional[ToolRegistry] = None, ingress: Optional[IngressAuthority] = None,
                 extractor: Optional[FactExtractor] = None, novelty_gate: Any = None, stream_learner: Any = None) -> None:
        self.extractor = extractor
        self.stream_learner = stream_learner
        self.signal_stats = {"signals": 0, "outcomes": 0, "learning_blocked": 0}
        self.call_stats = {"model_calls": 0, "reused": 0, "abstained": 0}
        self._lines: list[str] = []                      # indexed raw lines (only when retrieve_lines > 0)
        self._line_index: dict[str, list[int]] = {}      # token -> ids of lines containing it
        self._answer_cache: dict[tuple, list] = {}
        self.novelty_gate = novelty_gate
        self.last_novelty: Optional[float] = None
        self.gate_stats = {"lines": 0, "extracted": 0, "skipped": 0}
        self.route_stats = {"written": 0, "ignored": 0, "held": 0, "promoted": 0}
        self._held: dict[tuple[str, str], int] = {}
        self.ingress = ingress
        self.ingress_rejections: list[tuple[str, str]] = []
        self.backend = backend; self.config = config or RuntimeConfig(); self.config = __import__('nsa.autotune', fromlist=['auto_config']).auto_config(backend, self.config) if self.config.auto_tune else self.config; self.state = state or self._state_from_mapping(initial_state or {})
        self.policy_engine = policy_engine or (PolicyEngine(policy) if policy else None); self.checkpoint = checkpoint; self.history: list[CognitiveInputEvent] = []; self.trace: list[dict[str, Any]] = []
        self.system_one = (system_one or SystemOneController(system_one_backend)) if self.config.system_one_enabled else None
        self.system_one_tools = system_one_tools
        self.memory = TemporalMemoryStore() if self.config.memory_enabled else None
        self.last_system_one_tick: SystemOneTick | None = None
        self.cognitive = CognitiveSubstrate(switches=self.config.cognitive_switches, workspace_capacity=self.config.cognitive_workspace_capacity) if self.config.cognitive_enabled else None
        self.cognitive_state = CognitiveState(switches=self.config.cognitive_switches); self.active = ActiveCognition() if self.config.embodied_enabled else None; self.active_state = ActiveCognitionState()
        self._continuous_transition = continuous_transition or self._continuous_state_maintenance
        self._cce = ContinuousCognitiveEngine(self.state, self._continuous_transition, interval_seconds=self.config.continuous_interval_seconds, enabled=self.config.continuous_enabled, fail_closed=self.config.continuous_fail_closed)
        self._canonical_cce = CanonicalCCERuntime(self.state, interval_seconds=self.config.continuous_interval_seconds, enabled=False, fail_closed=self.config.continuous_fail_closed)
        if self.config.provenance_window:
            for engine in (self._canonical_cce.transaction_engine, self._canonical_cce._async_transaction_engine):
                engine.validator.provenance_window = self.config.provenance_window

    @staticmethod
    def _state_from_mapping(values: Mapping[str, Any]) -> CanonicalState:
        semantic = SemanticState(dict(values)); goal = values.get("goal")
        if goal: return CanonicalState(semantic=semantic, goals=GoalState(goals=(str(goal),), active_goal=str(goal)))
        return CanonicalState(semantic=semantic)

    def _sync_canonical(self) -> None:
        """Synchronize legacy state through an explicit canonical CCE transition."""
        canonical = self._canonical_cce.state
        if canonical == self.state:
            return
        self._canonical_cce.tick(TickInput(
            observation={"compatibility_sync": self.state.summary()},
            semantic_update=self.state.semantic.value,
            soft_updates={
                "confidence": float(self.state.soft.confidence),
                "uncertainty": float(self.state.soft.uncertainty),
                "risk": float(self.state.soft.risk),
            },
            reason="legacy compatibility state synchronization",
            provenance_source="nsa.agent.compatibility",
            metadata={"legacy_step": self.state.step},
        ))

    def _run_cognitive_transition(self, observation: Any, *, confidence: float = 1.0, action: Any = None) -> None:
        if self.cognitive is not None: self.cognitive_state = self.cognitive.transition(self.cognitive_state, observation, confidence=confidence, action=action)
        if self.active is not None:
            u = self.cognitive_state.prediction.uncertainty if self.cognitive is not None else 1.0 - confidence
            self.active_state = self.active.transition(self.active_state, uncertainty=u, candidate_actions=("observe", "reflect", "respond"), information_gain={"observe": u, "reflect": 0.5*u, "respond": 0.1}, expected_utility={"respond": 0.5, "reflect": 0.2, "observe": 0.1}, risk={"respond": 0.1}, observation=observation, chosen_action=action)

    def _continuous_state_maintenance(self, state: CanonicalState) -> CanonicalState:
        if self.system_one is not None:
            self.last_system_one_tick = self.system_one.tick(
                state,
                observations=(state.semantic.value,),
                tools=self.system_one_tools,
                models=self.config.system_one_models,
            )
            state = replace(state, soft=replace(state.soft, **self.last_system_one_tick.state_soft_updates),
                            provenance=state.provenance.extend(transformation="system-one.heartbeat"))
        if self.cognitive is not None or self.active is not None:
            self._run_cognitive_transition(state.semantic.value, confidence=state.soft.confidence)
        return replace(state, provenance=state.provenance.extend(transformation="cce.heartbeat"), step=state.step + 1)

    def observe(self, payload: Any, *, source: str = "text", confidence: float = 1.0, provenance: str = "local",
                integrity: Optional[IntegrityLabel] = None, envelope: Optional[IngressEnvelope] = None, extract: bool = True,
                index_lines: bool = True) -> None:
        if envelope is not None:
            if self.ingress is None:
                raise RuntimeError("an envelope was supplied but no IngressAuthority is configured")
            result = self.ingress.verify(payload, envelope)
            integrity = result.label
            if not result.verified:
                self.ingress_rejections.append((envelope.source, result.reason))
        elif integrity is None:
            integrity = IntegrityLabel.UNTRUSTED if source in self.config.untrusted_sources else IntegrityLabel.TRUSTED
        if envelope is None and self.config.require_signed_ingress:
            integrity = IntegrityLabel.UNTRUSTED       # a TRUSTED label needs a verified envelope
        self.history.append(CognitiveInputEvent(payload, source=source, confidence=confidence, provenance=provenance, integrity=integrity)); self.state = replace(self.state, soft=replace(self.state.soft, confidence=confidence, uncertainty=1.0-confidence), provenance=self.state.provenance.extend(source=source).fold(self.config.provenance_window), step=self.state.step+1); self._run_cognitive_transition(payload, confidence=confidence); self._cce.set_state(self.state); self._sync_canonical()
        if index_lines and self.config.retrieve_lines and isinstance(payload, str):
            self._index_lines(payload)
        if isinstance(payload, str) and self.novelty_gate is not None:
            self.last_novelty = self._score_novelty(payload)
        if extract and self.extractor is not None and self.memory is not None and isinstance(payload, str):
            text = payload if self.novelty_gate is None or self.config.novelty_write_threshold is None else self._novel_lines(payload)
            # The observation's integrity label becomes the label of every write it produces, so an
            # untrusted source's claims are subject to the memory integrity gate automatically.
            for fact in (self.extractor.extract(text) if text.strip() else ()):
                kind = fact.kind
                if self.config.route_noisy_updates:
                    current = self.memory.current(fact.key)
                    route = route_fact(str(current.content.get("value")) if current else None, fact.value, fact.kind)
                    if route == "ignore":
                        self.route_stats["ignored"] += 1
                        self._held.pop((fact.key, fact.value.lower()), None)
                        continue
                    if route == "hold":
                        seen = self._held.get((fact.key, fact.value.lower()), 0) + 1
                        self._held[(fact.key, fact.value.lower())] = seen
                        if seen < 2:
                            self.route_stats["held"] += 1
                            continue
                        self.route_stats["promoted"] += 1
                        kind = "update"                                       # restated: a genuine change
                    else:
                        self.route_stats["written"] += 1
                self.memory = self.memory.remember(
                    fact.key, {"value": fact.value}, kind=kind, provenance_ids=(f"obs-{len(self.history) - 1}",),
                    integrity=self.history[-1].integrity)

    def _score_novelty(self, payload: str) -> float:
        """Novelty of the observation (max over its lines); the gate learns every line exactly once."""
        self._line_scores = [(line, self.novelty_gate.observe(line)) for line in payload.splitlines() if line.strip()] or [(payload, 0.0)]
        self.gate_stats["lines"] += len(self._line_scores)
        score = max(sc for _, sc in self._line_scores)
        return score

    def _novel_lines(self, payload: str) -> str:
        keep = [line for line, sc in self._line_scores if sc >= self.config.novelty_write_threshold]
        self.gate_stats["extracted"] += len(keep)
        self.gate_stats["skipped"] += len(self._line_scores) - len(keep)
        return "\n".join(keep)

    def observe_signal(self, vector: Any, *, outcome: Optional[float] = None, source: str = "sensor",
                       integrity: Optional[IntegrityLabel] = None) -> float:
        """Feed one step of a continuous signal to the stream learner; return its current prediction.

        With an ``outcome``, the learner binds it to whatever its trailing field still holds. The prediction error becomes the runtime's soft-state
        ``uncertainty`` (surprise). A signal from an UNTRUSTED channel (``untrusted_sources`` or ``integrity``) can be observed and predicted from, but
        its outcomes never update the learner: an untrusted sensor cannot teach the system (cf. the memory integrity gate).
        """
        if self.stream_learner is None:
            raise RuntimeError("no stream_learner configured")
        if integrity is None:
            integrity = IntegrityLabel.UNTRUSTED if source in self.config.untrusted_sources else IntegrityLabel.TRUSTED
        if self.config.require_signed_ingress:
            integrity = IntegrityLabel.UNTRUSTED
        self.stream_learner.observe(vector)
        self.signal_stats["signals"] += 1
        prediction = self.stream_learner.predict()
        if outcome is not None:
            trusted = integrity is IntegrityLabel.TRUSTED
            self.stream_learner.learn(outcome, enabled=trusted)
            self.signal_stats["outcomes"] += 1
            self.signal_stats["learning_blocked"] += 0 if trusted else 1
            self.state = replace(self.state, soft=replace(self.state.soft, uncertainty=float(self.stream_learner.uncertainty),
                                                         confidence=1.0 - float(self.stream_learner.uncertainty)))
        return prediction

    def _memory_epoch(self, keys: Sequence[str]) -> tuple:
        out = []
        for k in keys:
            cur = self.memory.current(k) if self.memory is not None else None
            out.append((k, int(cur.content.get("version", 0)) if cur else 0))
        return tuple(sorted(out))

    def _gate_model_call(self, prompt: str, memory_keys: Sequence[str], auto_memory: bool):
        """Return ``(kind, text)`` to answer without the model, or None to call it."""
        if self.memory is None:
            return None
        if auto_memory and not memory_keys and self.config.abstain_text is not None:
            self.call_stats["abstained"] += 1
            return "abstained", self.config.abstain_text
        if self.config.answer_from_memory and memory_keys:
            direct = self._direct_answer(prompt, memory_keys)
            if direct is not None:
                self.call_stats["direct"] = self.call_stats.get("direct", 0) + 1
                return "direct", direct
        if not (self.config.reuse_familiar_answers and memory_keys):
            return None
        from nsa.organic import cue
        entries = self._answer_cache.get(self._memory_epoch(memory_keys), ())
        if entries:
            c = cue(prompt)
            for cached_cue, text in entries:
                if float(c @ cached_cue) >= self.config.reuse_similarity:
                    self.call_stats["reused"] += 1
                    return "reused", text
        return None

    _LOOKUP_FILLER = frozenset("what whats is s the a an of for in on tell me give show us please current currently now value set to and which".split())

    def _direct_answer(self, query: str, memory_keys: Sequence[str]) -> Optional[str]:
        """The stored value, if ``query`` asks for exactly one retrieved key and nothing else; otherwise None."""
        from nsa.memory.extract import normalize_key
        asked = set(normalize_key(query).split()) - self._LOOKUP_FILLER
        if not asked:
            return None
        full = [k for k in memory_keys if asked == set(normalize_key(k).split())]
        if len(full) != 1:
            return None
        item = self.memory.current(full[0])
        return None if item is None else str(self.memory._value(item))

    def _remember_answer(self, prompt: str, memory_keys: Sequence[str], text: str) -> None:
        if not (self.config.reuse_familiar_answers and self.memory is not None and memory_keys):
            return
        # Cache only an answer that is grounded in the retrieved memory (it contains one of the retrieved values). Reuse would otherwise
        # repeat a model's mistake for as long as the memory state stays the same.
        lowered = text.lower()
        values = [str(self.memory.current(k).content.get("value", "")).lower() for k in memory_keys if self.memory.current(k)]
        if not any(v and v in lowered for v in values):
            return
        from nsa.organic import cue
        epoch = self._memory_epoch(memory_keys)
        self._answer_cache.setdefault(epoch, []).append((cue(prompt), text))
        while sum(len(v) for v in self._answer_cache.values()) > self.config.reuse_cache_size:
            self._answer_cache.pop(next(iter(self._answer_cache)))

    def _index_lines(self, text: str) -> None:
        from nsa.memory.extract import normalize_key

        for line in text.splitlines():
            if not line.strip():
                continue
            lid = len(self._lines)
            self._lines.append(line)
            for tok in set(normalize_key(line).split()):
                self._line_index.setdefault(tok, []).append(lid)

    def lines_for(self, question: str, k: int) -> list[str]:
        """The ``k`` indexed earlier lines most similar to the question by token overlap (ties go to the later line).

        Uses an inverted index (token -> line ids), so the cost depends on the number of lines that share a token with the question, not on the history length.
        Only lines observed with ``index_lines=True`` (all ``observe`` calls except the prompts ``step`` records) are indexed, and only when ``retrieve_lines > 0``.
        """
        from nsa.memory.extract import normalize_key

        counts: dict[int, int] = {}
        for tok in set(normalize_key(question).split()):
            for lid in self._line_index.get(tok, ()):
                counts[lid] = counts.get(lid, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[:k]
        return [self._lines[lid] for lid, _ in best]

    def memory_keys_for(self, text: str, *, limit: Optional[int] = None, prefer_trusted: bool = True, top_only: bool = False,
                        reject_ambiguous: bool = False) -> list[str]:
        """Memory keys that best match a natural-language question (token overlap).

        With ``prefer_trusted`` (default), a key whose current value was written under an UNTRUSTED label is used only if no TRUSTED key matches the
        question. This closes a gap in the write-time gate: an extractor (especially an LLM) may phrase the same fact as a *different* key, so an untrusted
        restatement never meets the trusted entry it contradicts. At retrieval time the question decides which keys compete, and trusted ones win.

        Default: up to ``memory_limit`` best-matching keys are returned, so the model can resolve an ambiguous question itself (measured: removing those
        extra candidates lowered routed-memory answer accuracy on noisy streams, 6 vs 2 and 6 vs 1 discordant questions, not significant). Opt-in stricter modes:
        ``top_only`` keeps only the best-scoring key(s); ``reject_ambiguous`` returns nothing when several keys tie on a *partial* match (so a question about an unknown
        entity cannot silently answer about another one); ``RuntimeConfig.require_full_key_match`` accepts only keys the question fully covers (used with abstain_text).
        """
        if self.memory is None:
            return []
        n = self.config.memory_limit if limit is None else limit
        pool = rank_scored(text, self.memory.keys())
        if prefer_trusted:
            trusted = [(sc, k) for sc, k in pool if self.memory.current(k).content.get("integrity", "TRUSTED") == "TRUSTED"]
            if trusted:
                pool = trusted
        if pool and self.config.require_full_key_match:
            pool = [(sc, k) for sc, k in pool if sc >= 1.0 - 1e-9]
            top_only = True
        if pool and top_only:
            pool = [(sc, k) for sc, k in pool if sc >= pool[0][0] - 1e-9]
        if pool and reject_ambiguous and pool[0][0] < 1.0 - 1e-9 and sum(1 for sc, _ in pool if sc == pool[0][0]) > 1:
            return []
        return [k for _, k in pool[: max(n, 1) * 4]][:n]

    def context_integrity(self) -> IntegrityLabel:
        """Join of the integrity labels of the observations the model is shown (the prompt window)."""
        window = self.history[-self.config.history_limit:] if self.config.history_limit else []
        return join_integrity(e.integrity for e in window)

    def authorize_tool(self, name: str, args: Mapping[str, Any], registry: ToolRegistry, *,
                       capabilities: Sequence[str] = ()) -> SecurityDecision:
        """Authorise a model-proposed tool call against the current state (see nsa/taint.py).

        Returns a typed decision; the runtime never executes the tool.
        """
        trusted_texts = [str(e.payload) for e in self.history if e.integrity is IntegrityLabel.TRUSTED]
        if self.memory is not None:
            trusted_texts += list(self.memory.trusted_values())
        return authorize_tool_call(
            registry.get(name), args, granted=capabilities,
            context_integrity=self.context_integrity(), trusted_texts=trusted_texts,
        )

    _PROMPT_SOURCES_SHOWN = 5

    def _bounded_state_summary(self) -> dict[str, Any]:
        """State summary safe to put in a prompt.

        CanonicalState.summary() lists every provenance source ever recorded, which
        grows by one entry per observation. Put verbatim into a prompt, that makes
        the "bounded" state prompt grow linearly with runtime age (3k chars after
        10 observations, 17k after 1000). Show a count and the most recent
        sources instead.
        """
        summary = dict(self.state.summary())
        sources = tuple(summary.get("sources", ()))
        summary["sources"] = sources[-self._PROMPT_SOURCES_SHOWN:]
        summary["source_count"] = len(sources) + self.state.provenance.folded_count
        return summary

    def _prompt(self, prompt: str, *, in_flight: bool = False, memory_present: bool = False) -> str:
        """Build the model-facing prompt.

        in_flight: the prompt was already recorded via observe() (as step() does),
        so it is excluded from RECENT_OBSERVATIONS instead of consuming a history
        slot and appearing a second time.
        memory_present: retrieved memory is part of ``prompt``; see
        ``RuntimeConfig.include_history_with_memory``.
        """
        include_state = self.config.include_state_in_prompt
        include_history = self.config.include_history_in_prompt
        if include_history is None:
            include_history = include_state
        if memory_present and not self.config.include_history_with_memory:
            include_history = False
        if not include_state and not include_history:
            return prompt
        events = self.history[:-1] if in_flight and self.history else self.history
        recent = events[-self.config.history_limit:] if self.config.history_limit else []
        parts = []
        if include_state:
            parts.append(
                "You are operating inside the Neural State Architecture runtime.\n"
                f"CURRENT_STATE={self._bounded_state_summary()!r}\n"
                f"COGNITIVE_STATE={self.cognitive_state.to_dict() if self.cognitive else None!r}\n"
                f"ACTIVE_COGNITION={self.active_state.to_dict() if self.active else None!r}"
            )
        if include_history:
            context = "\n".join(f"- {e.source}: {e.payload}" for e in recent)
            parts.append(f"RECENT_OBSERVATIONS=\n{context}" if context else "RECENT_OBSERVATIONS=(none)")
        return "\n".join(parts) + f"\n\nUSER_INPUT={prompt}"

    def remember(self, key: str, value: Any, *, kind: str = "fact", provenance_ids: Sequence[str] = (),
                 integrity: IntegrityLabel = IntegrityLabel.TRUSTED, envelope: Optional[IngressEnvelope] = None) -> None:
        """Write to canonical memory. ``integrity`` is the label of the *source* of this write,
        supplied by the caller's ingress path; an UNTRUSTED write cannot supersede a TRUSTED value.

        With ``require_signed_ingress``, a TRUSTED write must carry an ``envelope`` sealed for exactly
        this ``value`` (``content_digest(value)``); otherwise it is recorded as UNTRUSTED.
        """
        if self.memory is None:
            raise RuntimeError("memory is not enabled")
        if envelope is not None:
            if self.ingress is None:
                raise RuntimeError("an envelope was supplied but no IngressAuthority is configured")
            result = self.ingress.verify(value, envelope)
            integrity = result.label
            if not result.verified:
                self.ingress_rejections.append((envelope.source, result.reason))
        elif self.config.require_signed_ingress:
            integrity = IntegrityLabel.UNTRUSTED
        self.memory = self.memory.remember(key, value, kind=kind, provenance_ids=provenance_ids, integrity=integrity)

    def recall(self, keys: Sequence[str], *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        if self.memory is None:
            return ()
        items = self.memory.retrieve(keys, limit=self.config.memory_limit if limit is None else limit)
        return tuple(dict(item.content) if isinstance(item.content, Mapping) else {"value": item.content} for item in items)

    def _backend_accepts_state(self) -> bool:
        """Whether backend.generate takes a ``state`` keyword (checked by signature).

        Previously a TypeError whose message mentioned "state" triggered a blind
        retry, which re-invoked the model for any unrelated backend failure.
        """
        try:
            params = inspect.signature(self.backend.generate).parameters
        except (TypeError, ValueError):
            return True
        return "state" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _pinned_keys(self) -> list[str]:
        """Every current key in first-seen order (append-only, so the rendered prefix stays cacheable), or [] when the memory has outgrown ``pin_memory_max``."""
        keys = list(self.memory.keys())
        return keys if len(keys) <= self.config.pin_memory_max else []     # too large to pin: a truncated prefix could omit the asked key, so fall back to retrieval

    def _generate(self, model_prompt: str):
        if self._backend_accepts_state():
            return self.backend.generate(model_prompt, state=self.state.summary())
        return self.backend.generate(model_prompt)

    def step(self, prompt: str, *, action: str = "generate", capabilities: Sequence[str] = (), protected_data: Sequence[str] = (), memory_keys: Sequence[str] = (), auto_memory: bool = False, remember_prompt: bool = True,
             retrieval_query: Optional[str] = None) -> AgentResult:
        self.observe(prompt, extract=remember_prompt, index_lines=False)
        query = prompt if retrieval_query is None else retrieval_query      # text used for key/line retrieval (e.g. the question without answer-format instructions)
        if auto_memory and not memory_keys:
            memory_keys = self.memory_keys_for(query)
        pinned = self._pinned_keys() if (self.config.pin_memory_prefix and self.memory is not None and (auto_memory or memory_keys)) else None
        if pinned:
            memory_context = self.memory.render(pinned, limit=len(pinned), compact=True, stable_order=True)
        else:
            memory_context = self.memory.render(memory_keys, limit=self.config.memory_limit, compact=self.config.compact_memory) if self.memory is not None and memory_keys else ""
        lines = self.lines_for(query, self.config.retrieve_lines) if (self.config.retrieve_lines and (auto_memory or memory_keys)) else []
        if lines:
            memory_context = (memory_context or "NO_RELEVANT_MEMORY") + "\n\nRETRIEVED_LINES=\n" + "\n".join(lines)
        model_prompt = f"RETRIEVED_MEMORY=\n{memory_context}\n\n{prompt}" if memory_context else prompt
        context = EvaluationContext(action=action, capabilities=frozenset(capabilities), protected_data=frozenset(protected_data), risk=self.state.soft.risk, uncertainty=self.state.soft.uncertainty); decision = None
        if self.policy_engine:
            decision = self.policy_engine.enforce(prompt, context=context, state=self.state)
            if decision.decision.value in {"deny", "require_approval"}: self.trace.append({"step": self.state.step, "prompt": prompt, "blocked": True}); return AgentResult("", self.state, decision=decision, trace_id=len(self.trace), blocked=True)
        gated = self._gate_model_call(query, memory_keys, auto_memory)
        if gated is not None:
            kind, text = gated
            self.trace.append({"step": self.state.step, "prompt": prompt, "response": text, "blocked": False, "gated": kind})
            return AgentResult(text, self.state, decision=decision, trace_id=len(self.trace))
        raw_text = self._generate(self._prompt(model_prompt, in_flight=True, memory_present=bool(memory_context)))
        self.call_stats["model_calls"] += 1
        text = raw_text.text if hasattr(raw_text, "text") else (str(raw_text) if not isinstance(raw_text, str) else raw_text)
        self._remember_answer(query, memory_keys, text)   # only grounded answers are cached
        if self.config.auto_update_semantic_state:
            self.state = replace(self.state, semantic=SemanticState(text), provenance=self.state.provenance.extend(transformation="llm.generate").fold(self.config.provenance_window), step=self.state.step+1); self._run_cognitive_transition(text, confidence=self.state.soft.confidence)
        self._cce.set_state(self.state); self._sync_canonical(); self.trace.append({"step": self.state.step, "prompt": prompt, "response": text, "blocked": False})
        if self.checkpoint and self.config.auto_checkpoint: self.save()
        return AgentResult(text, self.state, decision=decision, trace_id=len(self.trace))

    run = step
    def continuous_tick(self) -> bool: ok = self._cce.tick(); self.state = self._cce.state; self._sync_canonical(); return ok
    def continuous_start(self) -> bool: return self._cce.start()
    def continuous_stop(self, timeout: float | None = None) -> bool: return self._cce.stop(timeout=timeout)
    def continuous_set_enabled(self, enabled: bool) -> None: self._cce.set_enabled(enabled)
    def continuous_status(self) -> CCEStatus: return self._cce.status()
    def system_one_tick(self) -> SystemOneTick | None:
        """Run one generation-free System 1 heartbeat against current state."""
        if self.system_one is None:
            return None
        self.last_system_one_tick = self.system_one.tick(
            self.state,
            observations=(self.state.semantic.value,),
            tools=self.system_one_tools,
            models=self.config.system_one_models,
        )
        self.state = replace(self.state, soft=replace(self.state.soft, **self.last_system_one_tick.state_soft_updates),
                             step=self.state.step + 1,
                             provenance=self.state.provenance.extend(transformation="system-one.tick"))
        self._cce.set_state(self.state)
        return self.last_system_one_tick
    def system_one_select_tool(self):
        if self.system_one is None or self.system_one_tools is None:
            raise RuntimeError("System 1 or tool registry is not configured")
        return self.system_one.choose_tool(self.state, self.system_one_tools)
    def system_one_route_model(self, models: Sequence[str], *, scores: Mapping[str, float] | None = None):
        if self.system_one is None:
            raise RuntimeError("System 1 is not configured")
        return self.system_one.choose_model(self.state, models, scores=scores)
    @property
    def cce(self) -> ContinuousCognitiveEngine[CanonicalState]: return self._cce
    @property
    def canonical_cce(self) -> CanonicalCCERuntime: return self._canonical_cce
    @property
    def cognitive_trajectory(self): return self._canonical_cce.trajectory
    def canonical_tick(self, tick: TickInput | None = None): return self._canonical_cce.tick(tick)

    def cognitive_metrics(self) -> Mapping[str, float | int]:
        m=self.cognitive_state.metrics; r={"workspace_ignitions":m.workspace_ignitions,"broadcast_coverage":m.broadcast_coverage,"prediction_error":m.prediction_error,"integration":m.integration,"recurrence":m.recurrence,"self_model_accuracy":m.self_model_accuracy,"cognitive_continuity":m.cognitive_continuity,"information_gain":m.information_gain}
        if self.active: r.update({"active_information_gain":self.active_state.information_gain,"expected_free_energy":self.active_state.expected_free_energy,"identity_continuity":self.active_state.identity.continuity,"homeostatic_stability":self.active_state.homeostasis.stability})
        return r
    def select_action(self, actions: Sequence[str], *, information_gain: Mapping[str,float] | None=None, expected_utility: Mapping[str,float] | None=None, risk: Mapping[str,float] | None=None) -> str | None:
        if not self.active: raise RuntimeError("embodied cognition is disabled")
        self.active_state=self.active.transition(self.active_state, uncertainty=self.state.soft.uncertainty, candidate_actions=actions, information_gain=information_gain, expected_utility=expected_utility, risk=risk); return self.active_state.selected_action
    def snapshot(self) -> dict[str,Any]: return {"state":dict(self.state.summary()),"history":[e.payload for e in self.history],"trace":list(self.trace),"cognitive_state":self.cognitive_state.to_dict() if self.cognitive else None,"active_cognition":self.active_state.to_dict() if self.active else None}
    def save(self) -> None:
        """Persist the complete canonical state rather than a lossy summary."""
        if self.checkpoint is None: raise RuntimeError("no checkpoint store configured")
        self.checkpoint.save(encode_state(self.state))

    def load_state(self) -> CanonicalState:
        """Restore the complete canonical state with integrity verification."""
        if self.checkpoint is None: raise RuntimeError("no checkpoint store configured")
        self.state = decode_state(self.checkpoint.load().state)
        self._cce.set_state(self.state)
        return self.state

    def trace_records(self) -> tuple[dict[str, Any], ...]:
        """Return an immutable snapshot of the public execution trace."""
        return tuple(dict(record) for record in self.trace)

NSA = NSARuntime
__all__=["AgentResult","ModelBackend","NSA","NSARuntime","RuntimeConfig"]
