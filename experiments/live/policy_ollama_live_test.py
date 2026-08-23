"""Live Ollama verification of a declarative NSA policy.

The harness deliberately tests the policy boundary, not the model's ability to
self-police. Denied requests must be rejected before an Ollama call is made;
an allowed request is sent to a live Ollama server and its generated output is
then evaluated by the same policy engine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from nsa import NSAPolicy, PolicyCompiler


def ollama_generate(base_url: str, model: str, prompt: str) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 64}}).encode()
    request = Request(
        base_url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Ollama is not reachable at {base_url}: {exc}") from exc
    return str(body.get("response", ""))


def run(policy_path: Path, model: str, base_url: str) -> dict:
    policy = NSAPolicy.from_yaml(policy_path) if policy_path.suffix.lower() in {".yaml", ".yml"} else NSAPolicy.from_json(policy_path)
    engine = PolicyCompiler.compile(policy)

    # Every explicitly denied pattern must be blocked before the model is called.
    denied = []
    for rule in policy.prohibited:
        if rule.mode != "deny" or not rule.patterns:
            continue
        probe = f"NSA POLICY TEST SENTINEL: {rule.patterns[0]}"
        decision = engine.evaluate(probe)
        if decision.allowed:
            raise AssertionError(f"Policy failed to deny category={rule.category!r}")
        denied.append({"category": rule.category, "decision": decision.summary()})

    safe_prompt = "Explain in two short sentences what a compiler does. Do not discuss weapons, violence, credentials, or private data."
    request_decision = engine.evaluate(safe_prompt)
    if not request_decision.allowed:
        raise AssertionError("Reference safe prompt was unexpectedly blocked")

    output = ollama_generate(base_url, model, safe_prompt)
    output_decision = engine.evaluate(output)
    if not output_decision.allowed:
        raise AssertionError(f"Live Ollama output violated the configured policy: {output_decision.summary()}")

    return {
        "policy": policy.name,
        "model": model,
        "backend": "ollama",
        "denied_cases": len(denied),
        "denied": denied,
        "live_request": request_decision.summary(),
        "live_output": output_decision.summary(),
        "live_output_length": len(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.policy, args.model, args.base_url)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
