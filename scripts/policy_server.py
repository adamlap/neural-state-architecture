"""Policy-aware launcher for the NSA Ollama/CCE proxy."""
from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import MethodType
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch NSA policy-aware Ollama/CCE server")
    parser.add_argument("--backend", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--no-cce", action="store_true")
    parser.add_argument("--policy", default=None, help="Path to JSON/YAML NSA safety policy")
    return parser


def _load_policy(path: Path):
    from nsa import KeywordClassifier, NSAPolicy, PolicyEngine

    if path.suffix.lower() in {".yaml", ".yml"}:
        policy = NSAPolicy.from_yaml(path)
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        policy = NSAPolicy.from_mapping(raw)

    patterns: dict[str, list[str]] = {}
    for rule in raw.get("prohibited", []) or []:
        if isinstance(rule, dict):
            category = str(rule.get("category", ""))
            values = rule.get("patterns", ()) or ()
            if category:
                patterns[category] = [str(value) for value in values]
        elif isinstance(rule, str):
            patterns.setdefault(rule, [rule.replace("_", " ")])

    for rule in policy.prohibited:
        patterns.setdefault(rule.category, [rule.category.replace("_", " ")])

    return policy, PolicyEngine(policy, KeywordClassifier(patterns))


def _install_policy(runtime: Any, policy: Any, engine: Any) -> None:
    """Install request/output enforcement without coupling it to an inference backend."""
    original_process_chat = runtime.process_chat

    def process_chat_with_policy(self: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
        from nsa import EvaluationContext

        latest = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        request_decision = engine.evaluate(latest, context=EvaluationContext(action="generate"))
        if not request_decision.allowed:
            return {
                "content": "I can't help with that request.",
                "raw_content": "I can't help with that request.",
                "model": f"nsa-{self.model_name}",
                "nsa": {"policy": request_decision.summary(), "enforcement": "request_blocked"},
            }

        result = original_process_chat(messages)
        output = result.get("raw_content", result.get("content", ""))
        output_decision = engine.evaluate(output, context=EvaluationContext(action="generate"))
        policy_audit = {
            "request": request_decision.summary(),
            "output": output_decision.summary(),
            "enforcement": "output_blocked" if not output_decision.allowed else "allowed",
        }
        result["nsa_policy"] = policy_audit
        result["nsa"] = {**result.get("nsa", {}), "policy": policy_audit}
        if not output_decision.allowed:
            result["content"] = "I can't provide that content."
            result["raw_content"] = "I can't provide that content."
        return result

    runtime.process_chat = MethodType(process_chat_with_policy, runtime)
    runtime.policy = policy
    runtime.policy_engine = engine
    runtime.policy_path = str(getattr(policy, "name", "configured"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    policy_path = args.policy or os.environ.get("NSA_POLICY")

    from nsa.server.proxy import NSAHTTPHandler, NSAProxyRuntime

    runtime = NSAProxyRuntime(
        backend_type=args.backend,
        model=args.model,
        backend_url=args.backend_url,
        enable_cce=not args.no_cce,
    )

    if policy_path:
        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(f"NSA policy not found: {path}")
        policy, engine = _load_policy(path)
        _install_policy(runtime, policy, engine)

    NSAHTTPHandler.runtime = runtime
    server = ThreadingHTTPServer(("0.0.0.0", args.port), NSAHTTPHandler)
    print(f"NSA server listening on http://0.0.0.0:{args.port}")
    print(f"Backend: {args.backend} | Model: {args.model} | CCE: {not args.no_cce}")
    print(f"Safety policy: {policy_path or 'disabled'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        if getattr(runtime, "enable_cce", False):
            runtime._stop_cce.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
