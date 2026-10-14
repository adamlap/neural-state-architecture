"""Decoding wrapper so every benchmark condition is generated identically."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Generation:
    text: str
    prompt: str
    prompt_tokens: int | None
    completion_tokens: int | None
    timed_out: bool = False
    truncated: bool = False


@dataclass
class Completion:
    """Minimal stand-in for LLMGenerationOutput, returned when a call times out."""
    text: str = ""
    raw_response: dict = field(default_factory=dict)
    tokens: list = field(default_factory=list)


class FixedDecodingBackend:
    """Wrap a backend so each call uses the same temperature, token budget and context size.

    Without this, conditions routed through NSARuntime.step() fell back to the
    backend defaults (temperature 0.7, 256 tokens, a context size that depended on
    prompt length and on whether ``state`` was passed) while direct conditions used
    temperature 0 and 32 tokens. It also records the prompt that was actually sent,
    which differs from anything reconstructed afterwards (e.g. via agent._prompt()).
    """

    def __init__(self, inner: Any, *, temperature: float = 0.0, max_tokens: int = 32, num_ctx: int | None = 32768, retry_delay: float = 5.0):
        self.inner = inner
        self.model = getattr(inner, "model", getattr(inner, "model_name", "unknown"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.retry_delay = retry_delay
        self.last: Generation | None = None

    def generate(self, prompt: str, *, state: Any = None, **_ignored: Any):
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.num_ctx is not None:
            kwargs["num_ctx"] = self.num_ctx
        if state is not None:
            kwargs["state"] = state
        try:
            try:
                out = self.inner.generate(prompt, **kwargs)
            except RuntimeError as err:
                # A transient server fault (e.g. the Ollama runner restarting after a very
                # long prompt) answers HTTP 5xx. Retry once; a persistent fault still raises.
                if "http error 5" not in str(err).lower():
                    raise
                time.sleep(self.retry_delay)
                out = self.inner.generate(prompt, **kwargs)
        except RuntimeError as err:
            message = str(err).lower()
            if "timed out" in message or "timeout" in message:
                self.last = Generation("", prompt, None, None, timed_out=True)
                return Completion()
            raise
        raw = getattr(out, "raw_response", None) or {}
        text = out.text if hasattr(out, "text") else str(out)
        prompt_tokens = raw.get("prompt_eval_count") if isinstance(raw, dict) else None
        completion_tokens = raw.get("eval_count") if isinstance(raw, dict) else None
        truncated = bool(self.num_ctx and prompt_tokens is not None and prompt_tokens >= self.num_ctx - 1)
        self.last = Generation(text, prompt, prompt_tokens, completion_tokens, truncated=truncated)
        return out
