"""Backend protocols and simple adapters."""
from typing import Any


class OllamaBackend:
    """Thin Ollama adapter using its HTTP API."""

    def __init__(self, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str, *, state: dict[str, Any] | None = None) -> str:
        import urllib.request
        import json

        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read()) ["response"]
