"""Replaceable LLM backend adapters for the NSA runtime.

The core runtime never imports a vendor SDK.  Backends implement one small
``generate`` contract, which keeps the state/control plane model-agnostic.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional
from urllib import request


class BackendError(RuntimeError):
    """Raised when an LLM backend cannot complete an inference request."""


class OllamaBackend:
    """Zero-dependency adapter for the local Ollama HTTP API."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, state: Optional[Mapping[str, Any]] = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if state is not None:
            payload["options"] = {"num_ctx": 8192}
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised by integration tests
            raise BackendError(f"Ollama request failed: {exc}") from exc
        if "response" not in data:
            raise BackendError(f"Ollama response did not contain 'response': {data!r}")
        return str(data["response"])


class CallableBackend:
    """Adapter for a Python callable; useful for tests and custom inference stacks."""

    def __init__(self, fn, model: str = "callable") -> None:
        self.fn = fn
        self.model = model

    def generate(self, prompt: str, *, state: Optional[Mapping[str, Any]] = None) -> str:
        return str(self.fn(prompt, state=state))


class EchoBackend:
    """Deterministic backend for tests and examples."""

    model = "echo"

    def generate(self, prompt: str, *, state: Optional[Mapping[str, Any]] = None) -> str:
        return prompt


__all__ = ["BackendError", "CallableBackend", "EchoBackend", "OllamaBackend"]
