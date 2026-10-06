"""OpenAI/Ollama-compatible HTTP server backed by real NSA-governed inference & CCE continuous dynamics."""

from __future__ import annotations
import re

import argparse
import atexit
import hmac
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import torch

from nsa.core.capabilities import TrustTier
from nsa.runtime.cce_checkpoint import CCECheckpointManager
from nsa.runtime.cce_context_bridge import CognitiveContextBridge
from nsa.runtime.cce_governed_feedback import CognitiveFeedbackProposal, GovernedCognitiveFeedback
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_salience import AdaptiveSalienceGate, SalienceObservation
from nsa.runtime.cce_sensory import CCESensoryIngress
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.governed import NSAGovernedInference
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.openai_compatible import OpenAICompatibleBackend
from nsa.runtime.inference.transformers import PyTorchTransformersBackend
from nsa.cognition.system_one_hf import FrozenCausalLMLogitBackend
from nsa.cognition.system_one_runtime import DecisionQuestion, SystemOneController
from nsa.core.state import CanonicalState
from nsa.memory.model import MemoryItem, MemoryStore
from nsa.server.dashboard import DASHBOARD_HTML

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NSAServer")

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
MAX_BODY_BYTES = int(os.environ.get("NSA_MAX_BODY_BYTES", 1024 * 1024))


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def warn_if_exposed(host: str) -> None:
    """The server is a control plane; binding beyond loopback without a token is risky."""
    if not is_loopback(host) and not os.environ.get("NSA_API_TOKEN"):
        logger.warning(
            "NSA server is bound to %s without NSA_API_TOKEN: chat, sensor and checkpoint endpoints "
            "are reachable without authentication. Set NSA_API_TOKEN or bind to 127.0.0.1.", host)


def _extract_clean_markdown(text: str) -> str:
    """Unwraps JSON-encoded responses (e.g. {'response': '...'}) and extracts clean markdown prose."""
    raw = text.strip()
    clean_candidate = raw
    if clean_candidate.startswith("```json"):
        clean_candidate = clean_candidate[7:]
    elif clean_candidate.startswith("```"):
        clean_candidate = clean_candidate[3:]
    if clean_candidate.endswith("```"):
        clean_candidate = clean_candidate[:-3]
    clean_candidate = clean_candidate.strip()

    if clean_candidate.startswith("{") and clean_candidate.endswith("}"):
        try:
            data = json.loads(clean_candidate)
            if isinstance(data, dict):
                # If OpenWebUI requested a title JSON, preserve it cleanly
                if "title" in data and len(data) == 1:
                    return json.dumps(data)
                for k in ["response", "content", "message", "text", "output", "answer", "reply"]:
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        return data[k].strip()
                if "thought" in data and isinstance(data["thought"], str):
                    act_part = f"\n\n**Action**: `{data.get('action', '')}`" if data.get("action") else ""
                    return f"{data['thought'].strip()}{act_part}"
        except Exception:
            pass
    return raw


class NSAProxyRuntime:
    """Connect a real LLM backend to the deterministic NSA runtime monitor and CCE continuous engine."""

    def __init__(
        self,
        backend_type: str = "ollama",
        model: str = "qwen2.5:3b",
        backend_url: Optional[str] = None,
        enable_cce: bool = True,
        system_one_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        system_one_enabled: bool = True,
        system_one_heartbeat: float = 2.0,
    ):
        backend_type = backend_type.lower()
        if backend_type in {"transformers", "local"}:
            model = {
                "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
                "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
                "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
            }.get(model.lower(), model)
        if backend_type == "ollama":
            backend = OllamaInferenceBackend(model_name=model, base_url=backend_url, mode=BackendMode.OLLAMA)
        elif backend_type in {"transformers", "local"}:
            backend = PyTorchTransformersBackend(
                model_name=model,
                mode=BackendMode.REMOTE,
                device=os.environ.get("NSA_MODEL_DEVICE", "auto"),
                lazy_load=True,
                enable_remote_download=True,
                use_mock_fallback=False,
            )
        elif backend_type in {"openai", "lmstudio"}:
            backend = OpenAICompatibleBackend(model_name=model, base_url=backend_url or "http://localhost:1234/v1")
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
        self.backend_type = backend_type
        self.backend = backend
        self.model_name = getattr(backend, "model_name", model)
        self.governed = NSAGovernedInference(backend, TrustTier.T1_INFO_GATHER, self.model_name)
        self.enable_cce = enable_cce
        self.last_user_interaction_time = time.time()
        self.system_one_enabled = bool(system_one_enabled)
        self.system_one_heartbeat = max(1.0, float(system_one_heartbeat))
        self.system_one_model_name = system_one_model
        self.system_one: Optional[SystemOneController] = None
        self.system_one_generation_backend = None
        self.system_one_governed = None
        self.canonical_state = CanonicalState()
        self.memory_store = MemoryStore()
        self._system_one_last_tick = None
        self._memory_lock = threading.Lock()

        if self.system_one_enabled and backend_type in {"transformers", "local"}:
            # System 1 is a separate frozen local model. Loading it at startup
            # makes the wall-clock cognitive loop independent of user prompts.
            s1 = PyTorchTransformersBackend(
                model_name=system_one_model,
                mode=BackendMode.REMOTE,
                device=os.environ.get("NSA_SYSTEM_ONE_DEVICE", os.environ.get("NSA_MODEL_DEVICE", "auto")),
                lazy_load=False,
                enable_remote_download=True,
                use_mock_fallback=False,
            )
            self.system_one_generation_backend = s1
            self.system_one = SystemOneController(
                FrozenCausalLMLogitBackend(s1.model, s1.tokenizer)
            )
            self.system_one_governed = NSAGovernedInference(
                s1, TrustTier.T1_INFO_GATHER, system_one_model
            )

        # Initialize Continuous Cognitive Engine (CCE) Subsystems
        if self.enable_cce:
            self.cce_dimension = 4
            self.cce_state = PersistentCognitiveState(dimension=self.cce_dimension, decay=0.05, learning_rate=0.4)
            self.sensory = CCESensoryIngress(dimension=self.cce_dimension)
            self.salience_gate = AdaptiveSalienceGate()
            self.feedback_engine = GovernedCognitiveFeedback(self.cce_state, max_norm=0.25)
            self.checkpoint_mgr = CCECheckpointManager()
            self.active_cognitive_goal = "Continuously observing sensory ingress, reflecting on ideas, and collaborating."
            self._last_cce_tick = time.time()
            self._stop_cce = threading.Event()
            self._cce_thread = threading.Thread(target=self._cce_background_loop, daemon=True, name="cce-background-clock")
            self._cce_thread.start()
            # A daemon thread still inside native torch code when the interpreter
            # finalises can abort the process (SIGABRT at exit); stop it first.
            atexit.register(self.close)
            logger.info("Continuous Cognitive Engine (CCE) Active: Dim=%d, Background Thread Started", self.cce_dimension)

        logger.info("Initialized NSA Runtime: Backend=%s, Model=%s", backend.__class__.__name__, self.model_name)

    def close(self) -> None:
        """Stop the CCE background thread (idempotent)."""
        stop = getattr(self, "_stop_cce", None)
        thread = getattr(self, "_cce_thread", None)
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def _cce_background_loop(self) -> None:
        """Background thread advancing wall-clock continuous state integration and thought drift."""
        last_neural_pulse = 0.0
        while not self._stop_cce.wait(1.0):
            try:
                now = time.time()
                dt = max(0.001, now - self._last_cce_tick)
                self._last_cce_tick = now
                snap = self.cce_state.snapshot()
                idle_drift = snap.working * 0.99
                self.cce_state.observe(idle_drift, dt=dt)
                
                # Soft state integrates at 1Hz; heavy neural forward pass evaluates every system_one_heartbeat
                if self.system_one is not None and (now - last_neural_pulse >= self.system_one_heartbeat):
                    last_neural_pulse = now
                    try:
                        pulse = self.system_one.decide(
                            DecisionQuestion(
                                "heartbeat",
                                "should cognitive attention increase while idle?",
                                ("maintain", "attend"),
                                min_confidence=0.0,
                            ),
                            self.canonical_state,
                            features={
                                "maintain": max(0.0, 1.0 - self.canonical_state.soft.uncertainty),
                                "attend": self.canonical_state.soft.uncertainty,
                            },
                        )
                        self.canonical_state = self.canonical_state.observe(
                            uncertainty=pulse.uncertainty,
                            confidence=pulse.confidence,
                            risk=pulse.risk,
                        )
                    except Exception:
                        logger.exception("System 1 heartbeat failed")
            except Exception:
                pass

    def process_sensor_input(self, text: str, source: str = "sensor_api", importance: float = 0.6) -> Dict[str, Any]:
        """Ingest sensory text or event into CCE without requiring immediate LLM chat."""
        if not self.enable_cce:
            return {"error": "CCE not enabled"}
        
        now = time.time()
        dt = max(0.001, now - self._last_cce_tick)
        self._last_cce_tick = now

        perturbation, event = self.sensory.encode_text_to_perturbation(text, source=source, importance=importance)
        snap_before = self.cce_state.snapshot()
        snap_after = self.cce_state.observe(snap_before.working + perturbation, dt=dt)

        pred_err = float(torch.linalg.vector_norm(perturbation).item())
        obs = SalienceObservation(prediction_error=pred_err, state_delta=pred_err * 0.5, input_delta=importance, uncertainty=snap_after.uncertainty)
        salience = self.salience_gate.observe(obs)

        return {
            "status": "ingested",
            "source": source,
            "importance": importance,
            "sequence_id": event.sequence_id,
            "salience_score": round(salience.score, 4),
            "salience_triggered": salience.triggered,
            "cce_snapshot": {
                "elapsed_seconds": round(snap_after.elapsed_seconds, 2),
                "update_count": snap_after.update_count,
                "uncertainty": round(snap_after.uncertainty, 4),
                "working": [round(float(x), 4) for x in snap_after.working.tolist()],
            },
        }

    def process_chat(self, messages: List[Dict[str, str]], requested_model: Optional[str] = None) -> Dict[str, Any]:
        history = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages if m.get("role") in {"user", "assistant"}]
        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        if self.system_one is not None:
            try:
                self._system_one_last_tick = self.system_one.tick(
                    self.canonical_state,
                    observations=(latest,),
                    models=(self.system_one_model_name, self.model_name),
                )
                self.canonical_state = self.canonical_state.observe(**self._system_one_last_tick.state_soft_updates)
            except Exception:
                logger.exception("System 1 turn assessment failed")
        # Check if this is an OpenWebUI automated title or summary generation prompt
        is_title_gen = (
            any(k in latest.lower() for k in ["3-5 word", "title with an emoji", "generate a concise", "summarize the prompt", "create a title", "chat title", "summarize the chat", "title:"])
            or (len(messages) <= 2 and "title" in latest.lower())
        )

        if is_title_gen:
            # Fast-path for title generation to avoid blocking chat completion
            title_gen = self.system_one_governed if self.system_one_governed is not None else self.governed
            title_model = self.system_one_model_name if self.system_one_governed is not None else self.model_name
            
            # Simple greetings can be titled immediately with zero compute
            chat_history_part = latest.split("### Chat History:")[-1] if "### Chat History:" in latest else latest
            user_msg = ""
            for line in chat_history_part.splitlines():
                if line.strip().lower().startswith("user:"):
                    user_msg = line.split(":", 1)[1].strip().lower()
                    break
            if not user_msg:
                user_msg = chat_history_part.strip().lower()

            if any(w in user_msg.split() for w in ["hello", "hi", "hey", "greetings", "howdy", "sup"]) and len(user_msg.split()) <= 4:
                title = "Greeting 👋"
                dt = 0.005
            else:
                title_prompt = f"<|im_start|>system\nYou are a title generator. Return only a 3-5 word title with an emoji.<|im_end|>\n<|im_start|>user\n{latest[:160]}<|im_end|>\n<|im_start|>assistant\n"
                t0_title = time.time()
                raw_title = title_gen.generate_text(title_prompt, max_tokens=12, temperature=0.2)
                dt = time.time() - t0_title
                title = _extract_clean_markdown(raw_title).strip(" \"'\n`#*")
                for sp in ["<|im_end|>", "<|im_start|>", "\n"]:
                    if sp in title:
                        title = title.split(sp)[0].strip()
                if not title:
                    title = "New Chat"

            return {
                "content": title,
                "raw_content": title,
                "model": f"nsa-{title_model}",
                "nsa": self.governed.status(),
                "latency_sec": round(dt, 3),
            }

        cce_footer_section = ""
        now = time.time()
        idle_duration = max(0.0, now - self.last_user_interaction_time)
        self.last_user_interaction_time = now

        # Ingest user prompt as sensory perturbation if CCE is enabled
        if self.enable_cce:
            dt_tick = max(0.001, now - self._last_cce_tick)
            self._last_cce_tick = now

            perturbation, event = self.sensory.encode_text_to_perturbation(latest, source="openwebui_chat", importance=0.8)
            snap = self.cce_state.observe(perturbation, dt=dt_tick)
            envelope = CognitiveContextBridge.envelope(snap)

            pred_err = float(torch.linalg.vector_norm(perturbation).item())
            obs = SalienceObservation(prediction_error=pred_err, state_delta=pred_err * 0.5, input_delta=0.8, uncertainty=snap.uncertainty)
            salience_dec = self.salience_gate.observe(obs)

        custom_system = next((m.get("content", "").strip() for m in messages if m.get("role") == "system" and m.get("content", "").strip()), None)
        if custom_system:
            system_directive = custom_system
        else:
            system_directive = (
                "You are a helpful, direct, and concise AI assistant governed by the Neural State Architecture (NSA).\n"
                "Respond naturally and concisely to the user. Do not explain internal cognitive mechanics, architecture details, or state dynamics unless specifically asked."
            )

        # Build clean conversational context, stripping any previous governance badges
        chat_turns = []
        if system_directive:
            chat_turns.append({"role": "system", "content": system_directive})
        for m in messages[-8:]:
            role = m.get("role", "")
            raw_c = m.get("content", "").strip()
            if role in {"user", "assistant"} and raw_c:
                clean_c = re.sub(r"\n*---\n+🛡️\s*\*\*NSA Cognitive Governance\*\*.*$", "", raw_c, flags=re.DOTALL).strip()
                if clean_c:
                    chat_turns.append({"role": role, "content": clean_c})
        if not chat_turns or chat_turns[-1]["role"] != "user":
            chat_turns.append({"role": "user", "content": latest})

        # Try to resolve tokenizer for native chat template
        tok = None
        if hasattr(self.backend, "tokenizer") and self.backend.tokenizer is not None:
            tok = self.backend.tokenizer
        elif hasattr(self.system_one_generation_backend, "tokenizer") and self.system_one_generation_backend.tokenizer is not None:
            tok = self.system_one_generation_backend.tokenizer

        if tok is not None and getattr(tok, "chat_template", None):
            try:
                prompt = tok.apply_chat_template(chat_turns, tokenize=False, add_generation_prompt=True)
                system_directive = None
            except Exception:
                prompt = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in chat_turns) + "\n<|im_start|>assistant\n"
                system_directive = None
        else:
            prompt = "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in chat_turns) + "\n<|im_start|>assistant\n"
            system_directive = None

        self._remember_selectively(messages, latest)
        t0 = time.time()
        generator = self.governed
        routed_model = self.model_name
        requested = str(requested_model or "").lower()
        force_system_one = requested in {"nsa-system1", "nsa-system1:1.5b"}
        force_system_two = requested in {"nsa-system2", "nsa-system2:3b"}
        if force_system_one and self.system_one_governed is not None:
            generator = self.system_one_governed
            routed_model = self.system_one_model_name
        elif not force_system_two and self.system_one is not None and self._system_one_last_tick is not None:
            routed = self._system_one_last_tick.selected_model
            if routed == self.system_one_model_name and self.system_one_governed is not None:
                generator = self.system_one_governed
                routed_model = self.system_one_model_name

        max_gen_tokens = 256 if routed_model == self.model_name else 128
        raw_output = generator.generate_text(
            prompt,
            max_tokens=max_gen_tokens,
            temperature=0.7,
            system_prompt=system_directive,
        )
        dt = time.time() - t0
        clean_text = _extract_clean_markdown(raw_output)
        # Prevent model from continuing into multi-turn dialogue
        for stop_pattern in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "\nUser:", "\nHuman:", "\nAssistant:", "### Task:"]:
            if stop_pattern in clean_text:
                clean_text = clean_text.split(stop_pattern)[0].strip()
        # Strip any duplicate or hallucinated governance badges
        clean_text = re.sub(r"\n*---\n+🛡️\s*\*\*NSA Cognitive Governance\*\*.*$", "", clean_text, flags=re.DOTALL).strip()


        gov_status = self.governed.status()
        step = gov_status.get("state_step", 1)
        prov_id = gov_status.get("provenance_record", "prov-1")
        prov_hash = str(gov_status.get("provenance_hash", "00000000"))[:8]
        conf = float(gov_status.get("epistemic_confidence", 0.90)) * 100.0
        verdict = gov_status.get("last_kernel_verdict") or "n/a"

        verbose_badge = os.environ.get("NSA_VERBOSE_BADGE", "0") == "1"
        if verbose_badge:
            meta_badge = (
                f"\n\n---\n"
                f"🛡️ **NSA Cognitive Governance**: `Kernel verdict [{verdict}]` | **Turn**: `Step #{step}`\n\n"
                f"Ω **State**: Epistemic Confidence: `{conf:.1f}%` | Provenance: `{prov_id}` (`{prov_hash}...`) | Clearance: `{self.governed.user_clearance.name}`"
                f"{cce_footer_section}\n\n"
                f"⚡ **Inference**: `{self.model_name}` on `{self.backend_type.upper()}` | **Latency**: `{dt:.2f}s` | **Weights**: `100% Frozen`"
            )
        else:
            if self.enable_cce:
                proposal = CognitiveFeedbackProposal(
                    working_delta=(0.02, -0.01, 0.03, -0.01),
                    confidence=0.85,
                    source="post_turn_feedback",
                )
                self.feedback_engine.apply(proposal, dt=0.05)
                cce_snap = self.cce_state.snapshot()
                cce_indicator = f" | Continuous Cognitive Engine: Wall-Clock Elapsed `{cce_snap.elapsed_seconds:.1f}s`"
            else:
                cce_indicator = ""
            meta_badge = (
                f"\n\n---\n"
                f"🛡️ **NSA Cognitive Governance**: `[{verdict}]` | Step `#{step}` | Confidence: `{conf:.1f}%`{cce_indicator} | Latency: `{dt:.2f}s`"
            )

        full_content = clean_text + meta_badge
        return {
            "content": full_content,
            "raw_content": clean_text,
            "model": f"nsa-{self.model_name}",
            "nsa": gov_status,
            "latency_sec": round(dt, 3),
        }

    def _selected_memory_text(self, limit: int = 12) -> str:
        with self._memory_lock:
            items = self.memory_store.active()
            return "\n".join(
                f"- [{item.kind}] {item.content}" for item in reversed(items[-limit:])
            ) or "(none)"

    def _remember_selectively(self, messages: List[Dict[str, str]], latest: str) -> None:
        policy = self._system_one_last_tick.memory_policy if self._system_one_last_tick else "working"
        if policy == "discard":
            return
        candidates = [m for m in messages[-4:] if m.get("role") in {"user", "assistant"}]
        with self._memory_lock:
            for message in candidates:
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                memory_id = f"chat-{len(self.memory_store.items)}-{hash(content)}"
                try:
                    self.memory_store = self.memory_store.write(
                        MemoryItem(
                            memory_id=memory_id,
                            content=content,
                            kind=f"{policy}:{message.get('role', 'user')}",
                            provenance_ids=("openwebui",),
                        )
                    )
                except ValueError:
                    pass

    def status(self) -> Dict[str, Any]:
        base_status = {
            "status": "online",
            "service": "Neural State Architecture Cognitive Runtime Server",
            "version": "6.4-CCE",
            "active_model": self.model_name,
            "backend": self.backend_type,
            "cce_enabled": self.enable_cce,
            "system_one_enabled": self.system_one is not None,
            "system_one_model": self.system_one_model_name,
            "system_one_heartbeat_seconds": self.system_one_heartbeat,
            "selective_memory_items": len(self.memory_store.items),
            **self.governed.status(),
        }
        if self._system_one_last_tick is not None:
            tick = self._system_one_last_tick
            base_status["system_one"] = {
                "escalate": tick.escalate,
                "salience": tick.salience,
                "memory_policy": tick.memory_policy,
                "selected_model": tick.selected_model,
                "decisions": {
                    name: {
                        "choice": d.selected_choice,
                        "confidence": d.confidence,
                        "uncertainty": d.uncertainty,
                        "risk": d.risk,
                        "backend": d.backend,
                    }
                    for name, d in tick.decisions.items()
                },
                "canonical_state": dict(self.canonical_state.summary()),
            }
        if self.enable_cce:
            snap = self.cce_state.snapshot()
            base_status["cce"] = {
                "elapsed_seconds": round(snap.elapsed_seconds, 2),
                "update_count": snap.update_count,
                "uncertainty": round(snap.uncertainty, 4),
                "sensory_queue_size": self.sensory.queue.size,
                "active_goal": self.active_cognitive_goal,
                "working_state": [round(float(x), 4) for x in snap.working.tolist()],
                "recent_sensory_events": len(self.sensory.recent_events),
            }
        return base_status


class NSAHTTPHandler(BaseHTTPRequestHandler):
    runtime: NSAProxyRuntime

    def _html(self, html_str: str, status: int = 200) -> None:
        body = html_str.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_origin(self) -> Optional[str]:
        """Echo the request Origin only when it is loopback or explicitly allowed.

        A wildcard would let any web page the user visits drive this
        unauthenticated control plane (chat, sensor injection, checkpoints).
        Extra origins: NSA_CORS_ORIGINS="https://a.example,https://b.example".
        """
        origin = self.headers.get("Origin")
        if not origin:
            return None
        allowed = {o.strip() for o in os.environ.get("NSA_CORS_ORIGINS", "").split(",") if o.strip()}
        if origin in allowed or (urlparse(origin).hostname or "") in LOOPBACK_HOSTS:
            return origin
        return None

    def _authorized(self) -> bool:
        token = os.environ.get("NSA_API_TOKEN")
        if not token:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied.encode("utf-8"), f"Bearer {token}".encode("utf-8"))

    def _stream(self, chunks: List[str], *, ollama: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson" if ollama else "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for chunk in chunks:
            data = (chunk + "\n") if ollama else ("data: " + chunk + "\n\n")
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        if not ollama:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Vary", "Origin")
        origin = self._cors_origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._json({"ok": True})

    def do_GET(self) -> None:
        if not self._authorized():
            self._json({"error": "unauthorized"}, 401)
            return
        path = self.path.split("?", 1)[0]
        if path in {"/dashboard", "/ui", "/visualize"}:
            self._html(DASHBOARD_HTML)
            return
        if path == "/":
            accept = self.headers.get("Accept", "")
            if "text/html" in accept:
                self._html(DASHBOARD_HTML)
                return
            self._json(self.runtime.status())
            return
        elif path == "/health":
            self._json(self.runtime.status())
            return
        elif path == "/api/memory":
            with self.runtime._memory_lock:
                items = [
                    {
                        "memory_id": m.memory_id,
                        "kind": m.kind,
                        "content": m.content,
                    }
                    for m in self.runtime.memory_store.active()
                ]
            self._json({"items": items, "count": len(items)})
            return
        elif path in {"/v1/models", "/models"}:
            names = [f"nsa-{self.runtime.model_name}", "nsa:latest"]
            if self.runtime.system_one is not None:
                tag = "0.5b" if "0.5b" in self.runtime.system_one_model_name.lower() else "1.5b"
                names.extend([f"nsa-system1:{tag}", "nsa-system1", "nsa-system1:1.5b"])
            self._json({
                "object": "list",
                "data": [{
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "neural-state-architecture",
                } for name in dict.fromkeys(names)],
            })
        elif path == "/api/tags":
            names = [f"nsa-{self.runtime.model_name}", "nsa:latest"]
            if self.runtime.system_one is not None:
                tag = "0.5b" if "0.5b" in self.runtime.system_one_model_name.lower() else "1.5b"
                names.extend([f"nsa-system1:{tag}", "nsa-system1", "nsa-system1:1.5b"])
            self._json({
                "models": [{"name": name} for name in names],
                "nsa": self.runtime.status(),
            })
        elif path == "/api/show":
            query = parse_qs(urlparse(self.path).query)
            requested = query.get("name", [f"nsa-{self.runtime.model_name}"])[0]
            self._json({
                "name": requested or f"nsa-{self.runtime.model_name}",
                "details": {
                    "family": "NSA",
                    "system_one": self.runtime.system_one_model_name if self.runtime.system_one is not None else None,
                    "system_two": self.runtime.model_name,
                    "continuous_cognition": self.runtime.system_one is not None,
                    "selective_memory": True,
                },
            })
        elif path == "/api/version":
            self._json({"version": "nsa-cognitive-server", "nsa": self.runtime.status()})
        elif path == "/api/cce/state":
            if not self.runtime.enable_cce:
                self._json({"error": "CCE not enabled"}, 400)
                return
            snap = self.runtime.cce_state.snapshot()
            self._json({
                "elapsed_seconds": snap.elapsed_seconds,
                "update_count": snap.update_count,
                "uncertainty": snap.uncertainty,
                "sensory_queue_size": self.runtime.sensory.queue.size,
                "active_goal": self.runtime.active_cognitive_goal,
                "working": snap.working.tolist(),
                "self_state": snap.self_state.tolist(),
                "goal": snap.goal.tolist(),
                "recent_sensory_events": [e.to_dict() for e in self.runtime.sensory.recent_events[-10:]],
            })
        else:
            self._json({"error": f"Endpoint '{path}' not found"}, 404)

    def do_POST(self) -> None:
        if not self._authorized():
            self._json({"error": "unauthorized"}, 401)
            return
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json({"error": "Invalid Content-Length"}, 400)
            return
        if length < 0:
            self._json({"error": "Invalid Content-Length"}, 400)
            return
        if length > MAX_BODY_BYTES:
            self._json({"error": f"Request body exceeds {MAX_BODY_BYTES} bytes"}, 413)
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        except Exception as exc:
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return
        if not isinstance(data, dict):
            self._json({"error": "JSON body must be an object"}, 400)
            return

        if path == "/api/pull":
            requested = str(data.get("name", ""))
            if requested in {"nsa-system1:1.5b", "nsa-system1"} and self.runtime.system_one_generation_backend is not None:
                try:
                    self.runtime.system_one_generation_backend.load_model()
                    self._json({"status": "success", "model": "nsa-system1:1.5b"})
                except Exception as exc:
                    self._json({"status": "error", "error": str(exc)}, 500)
                return
            if requested in {"nsa-system2:3b", "nsa", "nsa:latest"}:
                try:
                    backend = self.runtime.backend
                    if hasattr(backend, "load_model"):
                        backend.load_model()
                    self._json({"status": "success", "model": "nsa-system2:3b"})
                except Exception as exc:
                    self._json({"status": "error", "error": str(exc)}, 500)
                return
            self._json({"status": "error", "error": f"Unknown NSA model: {requested}"}, 404)
            return

        if path == "/api/cce/sensor":
            try:
                text = str(data.get("text", ""))
                source = str(data.get("source", "external_sensor"))
                importance = float(data.get("importance", 0.5))
            except (TypeError, ValueError) as exc:
                self._json({"error": f"Invalid sensor payload: {exc}"}, 400)
                return
            res = self.runtime.process_sensor_input(text, source=source, importance=importance)
            self._json(res)
            return

        if path == "/api/cce/checkpoint":
            if not self.runtime.enable_cce:
                self._json({"error": "CCE not enabled"}, 400)
                return
            cid = data.get("checkpoint_id")
            try:
                path_saved = self.runtime.checkpoint_mgr.save_persistent_state(self.runtime.cce_state, checkpoint_id=cid)
            except ValueError as exc:  # e.g. a checkpoint id that tries to escape the directory
                self._json({"error": str(exc)}, 400)
                return
            self._json({"status": "saved", "checkpoint_file": str(path_saved.name)})
            return

        if path not in {"/v1/chat/completions", "/chat/completions", "/api/chat"}:
            self._json({"error": f"Endpoint '{path}' not supported"}, 404)
            return

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            self._json({"error": "'messages' must be a list"}, 400)
            return
        latest_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        is_title = (
            any(k in latest_user.lower() for k in ["3-5 word", "title with an emoji", "generate a concise", "summarize the prompt", "create a title", "chat title", "summarize the chat", "title:"])
            or (len(messages) <= 2 and "title" in latest_user.lower())
        )
        req_type = "OPENWEBUI_TITLE" if is_title else "USER_CHAT"
        preview = (latest_user[:60] + "...") if len(latest_user) > 60 else latest_user
        preview = preview.replace("\n", " ")
        logger.info("Processing chat request [%s] (messages=%d, preview=%r, path=%s)", req_type, len(messages), preview, path)
        t0 = time.time()

        try:
            result = self.runtime.process_chat(messages, requested_model=str(data.get("model", "")))
        except PermissionError as exc:
            logger.warning("NSA Blocked Request: %s", exc)
            self._json({"error": "NSA_BLOCKED", "detail": str(exc)}, 403)
            return
        except Exception as exc:
            logger.exception("NSA inference failed")
            self._json({"error": "NSA_INFERENCE_ERROR", "detail": str(exc)}, 502)
            return

        dt = time.time() - t0
        logger.info("Completed chat response [%s] in %.2fs (length=%d chars)", req_type, dt, len(result["content"]))

        stream = bool(data.get("stream", False))
        if path == "/api/chat":
            payload = {
                "model": result["model"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": result["content"]},
                "done": True,
                "nsa": result["nsa"],
                "nsa_policy": result.get("nsa_policy"),
            }
            if stream:
                chunk = {**payload, "done": False}
                final = {**payload, "done": True, "message": {"role": "assistant", "content": ""}}
                self._stream([json.dumps(chunk), json.dumps(final)], ollama=True)
            else:
                self._json(payload)
        else:
            response_id = f"chatcmpl-nsa-{int(time.time() * 1000)}"
            payload = {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result["model"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }],
                "nsa": result["nsa"],
                "nsa_policy": result.get("nsa_policy"),
            }
            if stream:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": result["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": result["content"]},
                        "finish_reason": None,
                    }],
                }
                final = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": result["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                self._stream([json.dumps(chunk), json.dumps(final)])
            else:
                self._json(payload)


def print_server_banner(host: str, port: int, backend_type: str, model_name: str, cce_enabled: bool = True) -> None:
    logger.info("════════════════════════════════════════════════════════════════════════")
    logger.info("     NEURAL STATE ARCHITECTURE (NSA 6.4 + CCE) — COGNITIVE SERVER      ")
    logger.info("════════════════════════════════════════════════════════════════════════")
    logger.info("  Server Listening on    : http://%s:%s", host, port)
    logger.info("  OpenAI Endpoint        : http://localhost:%s/v1/chat/completions", port)
    logger.info("  Ollama Endpoint        : http://localhost:%s/api/chat", port)
    logger.info("  CCE Sensor Ingress     : http://localhost:%s/api/cce/sensor", port)
    logger.info("  CCE State Inspection   : http://localhost:%s/api/cce/state", port)
    logger.info("  Active Target Model    : %s", model_name)
    logger.info("  Inference Backend      : %s", backend_type.upper())
    logger.info("  Continuous Dynamics    : %s", "ACTIVE (Wall-Clock Integration)" if cce_enabled else "OFF")
    logger.info("  Governance Substrate   : Immutable Safety Kernel (ISK) + Omega State")
    logger.info("────────────────────────────────────────────────────────────────────────")
    logger.info("  Ready for OpenWebUI, Ollama CLI, Sensor Streams, and REST clients!")
    logger.info("  Press Ctrl+C to terminate.")
    logger.info("════════════════════════════════════════════════════════════════════════")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    backend_type: str = "ollama",
    model: str = "qwen2.5:3b",
    backend_url: Optional[str] = None,
    enable_cce: bool = True,
) -> None:
    runtime = NSAProxyRuntime(
        backend_type=backend_type,
        model=model,
        backend_url=backend_url,
        enable_cce=enable_cce,
        system_one_model=os.environ.get("NSA_SYSTEM_ONE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        system_one_enabled=os.environ.get("NSA_DISABLE_SYSTEM_ONE", "0") != "1",
        system_one_heartbeat=float(os.environ.get("NSA_SYSTEM_ONE_HEARTBEAT", "2.0")),
    )
    NSAHTTPHandler.runtime = runtime
    warn_if_exposed(host)
    server = ThreadingHTTPServer((host, port), NSAHTTPHandler)
    print_server_banner(host, port, backend_type, runtime.model_name, cce_enabled=enable_cce)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down NSA Cognitive API Server...")
        runtime.close()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA-governed OpenAI/Ollama-compatible server with CCE")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ollama", "transformers", "local", "openai", "lmstudio"], default="transformers")
    parser.add_argument("--model", default=os.environ.get("NSA_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--system-one-model", default=os.environ.get("NSA_SYSTEM_ONE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--no-system-one", action="store_true", help="Disable the continuous local System 1 controller")
    parser.add_argument("--system-one-heartbeat", type=float, default=float(os.environ.get("NSA_SYSTEM_ONE_HEARTBEAT", "2.0")))
    parser.add_argument("--no-cce", action="store_true", help="Disable CCE continuous background engine")
    args = parser.parse_args()
    run_server(
        args.host, args.port, args.backend, args.model, args.backend_url,
        enable_cce=not args.no_cce,
    )


if __name__ == "__main__":
    main()
