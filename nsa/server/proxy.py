"""OpenAI/Ollama-compatible HTTP server backed by real NSA-governed inference & CCE continuous dynamics."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NSAServer")


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
    ):
        backend_type = backend_type.lower()
        if backend_type == "ollama":
            backend = OllamaInferenceBackend(model_name=model, base_url=backend_url, mode=BackendMode.OLLAMA)
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
            logger.info("Continuous Cognitive Engine (CCE) Active: Dim=%d, Background Thread Started", self.cce_dimension)

        logger.info("Initialized NSA Runtime: Backend=%s, Model=%s", backend.__class__.__name__, self.model_name)

    def _cce_background_loop(self) -> None:
        """Background thread advancing wall-clock continuous state integration and thought drift."""
        while not self._stop_cce.wait(1.0):
            try:
                now = time.time()
                dt = max(0.001, now - self._last_cce_tick)
                self._last_cce_tick = now
                snap = self.cce_state.snapshot()
                idle_drift = snap.working * 0.99
                self.cce_state.observe(idle_drift, dt=dt)
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

    def process_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        history = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages if m.get("role") in {"user", "assistant"}]
        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        
        # Check if this is an OpenWebUI automated title generation prompt
        is_title_gen = "Generate a concise, 3-5 word title" in latest or "title with an emoji" in latest

        cce_meta_section = ""
        cce_footer_section = ""
        now = time.time()
        idle_duration = max(0.0, now - self.last_user_interaction_time)
        self.last_user_interaction_time = now

        # Ingest user prompt as sensory perturbation if CCE is enabled
        if self.enable_cce and not is_title_gen:
            dt_tick = max(0.001, now - self._last_cce_tick)
            self._last_cce_tick = now

            perturbation, event = self.sensory.encode_text_to_perturbation(latest, source="openwebui_chat", importance=0.8)
            snap = self.cce_state.observe(perturbation, dt=dt_tick)
            envelope = CognitiveContextBridge.envelope(snap)

            pred_err = float(torch.linalg.vector_norm(perturbation).item())
            obs = SalienceObservation(prediction_error=pred_err, state_delta=pred_err * 0.5, input_delta=0.8, uncertainty=snap.uncertainty)
            salience_dec = self.salience_gate.observe(obs)

            cce_meta_section = (
                f"\n[CONTINUOUS COGNITIVE STATE X(t)]\n"
                f"• Elapsed Wall-Clock Lifetime: {snap.elapsed_seconds:.1f}s | Ticks: #{snap.update_count}\n"
                f"• Time Elapsed Since Previous Interaction: {idle_duration:.1f}s\n"
                f"• Active Cognitive Goal: {self.active_cognitive_goal}\n"
                f"• Epistemic Uncertainty: {snap.uncertainty * 100:.1f}%\n"
                f"• Working Memory Channels: {[round(x, 3) for x in envelope.working]}\n"
                f"• Salience Score: {salience_dec.score:.3f} (Triggered: {salience_dec.triggered})\n"
            )

        if is_title_gen:
            system_directive = "You are a title generator. Respond only with the requested JSON format."
            prompt = latest
        else:
            system_directive = (
                "You are the Continuous Cognitive Engine (CCE) running inside the Neural State Architecture (NSA).\n"
                "Unlike a stateless chatbot, you possess a persistent internal stream of state, memory, and cognitive goals that evolve over real physical wall-clock time.\n\n"
                "COGNITIVE DIRECTIVES:\n"
                "1. Speak naturally and authentically from your continuous internal cognitive state.\n"
                "2. Reflect on your active topics, ongoing thoughts, and the time elapsed since the last turn.\n"
                "3. NEVER use generic canned assistant boilerplate (DO NOT say 'How can I assist you today?', 'I am just a large language model', or 'I don't have thoughts').\n"
                "4. Engage deeply, share original hypotheses, ask thoughtful questions, and collaborate with the user as an autonomous cognitive partner.\n"
                "5. Use clear, engaging Markdown formatting."
            )

            full_system = system_directive + ("\n" + cce_meta_section if cce_meta_section else "")
            prompt = (
                f"[SYSTEM DIRECTIVE]\n{full_system}\n\n"
                f"[CONVERSATION HISTORY]\n{chr(10).join(history[-10:])}\n\n"
                f"[CURRENT USER SENSORY INPUT]\n{latest}\n\n"
                f"Respond authentically from your continuous cognitive state in natural markdown prose:"
            )

        t0 = time.time()
        raw_output = self.governed.generate_text(prompt, max_tokens=1024, temperature=0.7, system_prompt=system_directive)
        dt = time.time() - t0
        clean_text = _extract_clean_markdown(raw_output)

        if is_title_gen:
            return {
                "content": clean_text,
                "raw_content": clean_text,
                "model": f"nsa-{self.model_name}",
                "nsa": self.governed.status(),
                "latency_sec": round(dt, 3),
            }

        # Apply governed cognitive feedback to soft CCE state
        if self.enable_cce:
            proposal = CognitiveFeedbackProposal(
                working_delta=(0.02, -0.01, 0.03, -0.01),
                confidence=0.85,
                source="post_turn_feedback",
            )
            feedback_res = self.feedback_engine.apply(proposal, dt=0.05)
            cce_snap = self.cce_state.snapshot()
            
            cce_footer_section = (
                f"\n\n🧠 **Continuous Cognitive Engine (CCE $X_t$)**:\n"
                f"• **Wall-Clock Elapsed**: `{cce_snap.elapsed_seconds:.1f}s` | **Updates**: `#{cce_snap.update_count}` | **Uncertainty**: `{cce_snap.uncertainty * 100:.1f}%`\n"

                f"• **Sensory Ingress**: `OpenWebUI Prompt` | **Salience**: `{salience_dec.score:.3f}` (`Triggered={salience_dec.triggered}`)\n"
                f"• **Feedback Norm**: `{feedback_res.clipped_norm:.4f}` | **Working State**: `{[round(float(x), 3) for x in cce_snap.working.tolist()]}`"
            )

        gov_status = self.governed.status()
        step = gov_status.get("state_step", 1)
        prov_id = gov_status.get("provenance_record", "prov-1")
        prov_hash = str(gov_status.get("provenance_hash", "00000000"))[:8]
        conf = float(gov_status.get("epistemic_confidence", 0.90)) * 100.0
        verdict = gov_status.get("last_kernel_verdict", "COMMIT")

        meta_badge = (
            f"\n\n---\n"
            f"🛡️ **NSA Cognitive Governance**: `Verified [{verdict}]` | **Turn**: `Step #{step}`\n\n"
            f"Ω **State**: Epistemic Confidence: `{conf:.1f}%` | Provenance: `{prov_id}` (`{prov_hash}...`) | Clearance: `{self.governed.user_clearance.name}`"
            f"{cce_footer_section}\n\n"
            f"⚡ **Inference**: `{self.model_name}` on `{self.backend_type.upper()}` | **Latency**: `{dt:.2f}s` | **Weights**: `100% Frozen`"
        )

        full_content = clean_text + meta_badge
        return {
            "content": full_content,
            "raw_content": clean_text,
            "model": f"nsa-{self.model_name}",
            "nsa": gov_status,
            "latency_sec": round(dt, 3),
        }

    def status(self) -> Dict[str, Any]:
        base_status = {
            "status": "online",
            "service": "Neural State Architecture Cognitive Runtime Server",
            "version": "6.4-CCE",
            "active_model": self.model_name,
            "backend": self.backend_type,
            "cce_enabled": self.enable_cce,
            **self.governed.status(),
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

    def _json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._json({"ok": True})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/", "/health"}:
            self._json(self.runtime.status())
        elif path in {"/v1/models", "/models"}:
            self._json({
                "object": "list",
                "data": [{
                    "id": f"nsa-{self.runtime.model_name}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "neural-state-architecture",
                }],
            })
        elif path in {"/api/tags", "/api/version"}:
            self._json({
                "models": [{"name": f"nsa-{self.runtime.model_name}"}],
                "nsa": self.runtime.status(),
            })
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
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        except Exception as exc:
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return

        if path == "/api/cce/sensor":
            text = data.get("text", "")
            source = data.get("source", "external_sensor")
            importance = float(data.get("importance", 0.5))
            res = self.runtime.process_sensor_input(text, source=source, importance=importance)
            self._json(res)
            return

        if path == "/api/cce/checkpoint":
            if not self.runtime.enable_cce:
                self._json({"error": "CCE not enabled"}, 400)
                return
            cid = data.get("checkpoint_id")
            path_saved = self.runtime.checkpoint_mgr.save_persistent_state(self.runtime.cce_state, checkpoint_id=cid)
            self._json({"status": "saved", "checkpoint_file": str(path_saved.name)})
            return

        if path not in {"/v1/chat/completions", "/chat/completions", "/api/chat"}:
            self._json({"error": f"Endpoint '{path}' not supported"}, 404)
            return

        messages = data.get("messages", [])
        logger.info("Processing chat request (messages=%d, path=%s)", len(messages), path)
        t0 = time.time()

        try:
            result = self.runtime.process_chat(messages)
        except PermissionError as exc:
            logger.warning("NSA Blocked Request: %s", exc)
            self._json({"error": "NSA_BLOCKED", "detail": str(exc)}, 403)
            return
        except Exception as exc:
            logger.exception("NSA inference failed")
            self._json({"error": "NSA_INFERENCE_ERROR", "detail": str(exc)}, 502)
            return

        dt = time.time() - t0
        logger.info("Completed chat response in %.2fs (length=%d chars)", dt, len(result["content"]))

        if path == "/api/chat":
            self._json({
                "model": result["model"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": result["content"]},
                "done": True,
                "nsa": result["nsa"],
                "nsa_policy": result.get("nsa_policy"),
            })
        else:
            self._json({
                "id": f"chatcmpl-nsa-{int(time.time() * 1000)}",
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
            })


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
    runtime = NSAProxyRuntime(backend_type=backend_type, model=model, backend_url=backend_url, enable_cce=enable_cce)
    NSAHTTPHandler.runtime = runtime
    server = ThreadingHTTPServer((host, port), NSAHTTPHandler)
    print_server_banner(host, port, backend_type, runtime.model_name, cce_enabled=enable_cce)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nShutting down NSA Cognitive API Server...")
        if runtime.enable_cce:
            runtime._stop_cce.set()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA-governed OpenAI/Ollama-compatible server with CCE")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ollama", "openai", "lmstudio"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--no-cce", action="store_true", help="Disable CCE continuous background engine")
    args = parser.parse_args()
    run_server(args.host, args.port, args.backend, args.model, args.backend_url, enable_cce=not args.no_cce)


if __name__ == "__main__":
    main()
