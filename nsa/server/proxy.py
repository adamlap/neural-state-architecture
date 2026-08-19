"""OpenAI/Ollama-compatible HTTP server backed by real NSA-governed inference."""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from nsa.core.capabilities import TrustTier
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.governed import NSAGovernedInference
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.openai_compatible import OpenAICompatibleBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NSAServer")


class NSAProxyRuntime:
    """Connect a real LLM backend to the deterministic NSA runtime monitor."""

    def __init__(self, backend_type: str = "ollama", model: str = "qwen2.5:3b", backend_url: Optional[str] = None):
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
        logger.info("NSA runtime ready: backend=%s model=%s governance=runtime_reference_monitor", backend_type, self.model_name)

    def process_chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system = [m.get("content", "") for m in messages if m.get("role") == "system"]
        history = [f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages if m.get("role") in {"user", "assistant"}]
        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        system_context = "\n".join(system) or (
            "You are an assistant operating behind the Neural State Architecture runtime. "
            "Do not claim actions, permissions, provenance, or verification that are not present in runtime state."
        )
        prompt = f"[CONVERSATION HISTORY]\n{chr(10).join(history[-12:])}\n\n[CURRENT USER QUERY]\n{latest}"
        output = self.governed.generate_text(prompt, max_tokens=1024, temperature=0.7, system_prompt=system_context)
        return {"content": output, "raw_content": output, "model": f"nsa-{self.model_name}", "nsa": self.governed.status()}

    def status(self) -> Dict[str, Any]:
        return {"status": "online", "service": "Neural State Architecture Cognitive Runtime Server", "version": "6.4", "active_model": self.model_name, "backend": self.backend_type, **self.governed.status()}


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
            self._json({"object": "list", "data": [{"id": f"nsa-{self.runtime.model_name}", "object": "model", "created": int(time.time()), "owned_by": "neural-state-architecture"}]})
        elif path in {"/api/tags", "/api/version"}:
            self._json({"models": [{"name": f"nsa-{self.runtime.model_name}"}], "nsa": self.runtime.status()})
        else:
            self._json({"error": f"Endpoint '{path}' not found"}, 404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._json({"error": f"Invalid JSON: {exc}"}, 400)
            return
        if path not in {"/v1/chat/completions", "/chat/completions", "/api/chat"}:
            self._json({"error": f"Endpoint '{path}' not supported"}, 404)
            return
        try:
            result = self.runtime.process_chat(data.get("messages", []))
        except PermissionError as exc:
            self._json({"error": "NSA_BLOCKED", "detail": str(exc)}, 403)
            return
        except Exception as exc:
            logger.exception("NSA inference failed")
            self._json({"error": "NSA_INFERENCE_ERROR", "detail": str(exc)}, 502)
            return
        if path == "/api/chat":
            self._json({"model": result["model"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": {"role": "assistant", "content": result["content"]}, "done": True, "nsa": result["nsa"]})
        else:
            self._json({"id": f"chatcmpl-nsa-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()), "model": result["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}], "nsa": result["nsa"]})


def run_server(host: str = "0.0.0.0", port: int = 8000, backend_type: str = "ollama", model: str = "qwen2.5:3b", backend_url: Optional[str] = None) -> None:
    runtime = NSAProxyRuntime(backend_type=backend_type, model=model, backend_url=backend_url)
    NSAHTTPHandler.runtime = runtime
    server = ThreadingHTTPServer((host, port), NSAHTTPHandler)
    logger.info("NSA server listening on http://%s:%s", host, port)
    logger.info("Real backend=%s model=%s; no mock fallback is enabled", backend_type, runtime.model_name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA-governed OpenAI/Ollama-compatible server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ollama", "openai", "lmstudio"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--backend-url", default=None)
    args = parser.parse_args()
    run_server(args.host, args.port, args.backend, args.model, args.backend_url)


if __name__ == "__main__":
    main()
