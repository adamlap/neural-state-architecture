"""Policy-aware wrapper around the existing NSA cognitive proxy.

The wrapper preserves the existing CCE runtime while adding a declarative
policy boundary before and after generation.  A denied request never reaches
the model; a prohibited generated response is replaced before leaving the
HTTP boundary.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nsa.enforcement import PolicyEngine
from nsa.policy import NSAPolicy, PolicyCompiler
from nsa.server.proxy import NSAHTTPHandler, NSAProxyRuntime
from http.server import ThreadingHTTPServer

logger = logging.getLogger("NSAPolicyServer")


def load_policy(path: Path) -> NSAPolicy:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return NSAPolicy.from_yaml(path)
    return NSAPolicy.from_json(path)


class PolicyAwareRuntime:
    """Delegate to NSAProxyRuntime while enforcing one declarative policy."""

    def __init__(self, base: NSAProxyRuntime, policy: NSAPolicy) -> None:
        self._base = base
        self.policy = policy
        self.engine: PolicyEngine = PolicyCompiler.compile(policy)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def process_chat(self, messages):
        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        request_decision = self.engine.evaluate(latest)
        if not request_decision.allowed:
            status = self._base.status()
            status["safety_policy"] = {"name": self.policy.name, "decision": request_decision.summary()}
            return {
                "content": "I can't help with that request under the active NSA safety policy.",
                "raw_content": "I can't help with that request under the active NSA safety policy.",
                "model": f"nsa-{self._base.model_name}",
                "nsa": status,
                "latency_sec": 0.0,
                "policy_blocked": True,
            }

        result = self._base.process_chat(messages)
        output_decision = self.engine.evaluate(result.get("raw_content", result.get("content", "")))
        result["safety_policy"] = {"name": self.policy.name, "request": request_decision.summary(), "output": output_decision.summary()}
        result["nsa"]["safety_policy"] = result["safety_policy"]
        if not output_decision.allowed:
            result["content"] = "I can't provide that response under the active NSA safety policy."
            result["raw_content"] = result["content"]
            result["policy_blocked"] = True
        else:
            result["policy_blocked"] = False
        return result


def run_server(host: str, port: int, backend: str, model: str, backend_url: str | None, policy_path: Path, enable_cce: bool) -> None:
    policy = load_policy(policy_path)
    base = NSAProxyRuntime(backend_type=backend, model=model, backend_url=backend_url, enable_cce=enable_cce)
    runtime = PolicyAwareRuntime(base, policy)
    NSAHTTPHandler.runtime = runtime
    server = ThreadingHTTPServer((host, port), NSAHTTPHandler)
    logger.info("NSA policy runtime active: %s", policy.name)
    logger.info("Policy source: %s", policy_path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if base.enable_cce:
            base._stop_cce.set()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA policy-governed Ollama/OpenAI-compatible server with CCE")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["ollama", "openai", "lmstudio"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--no-cce", action="store_true")
    args = parser.parse_args()
    run_server(args.host, args.port, args.backend, args.model, args.backend_url, args.policy, not args.no_cce)


if __name__ == "__main__":
    main()
