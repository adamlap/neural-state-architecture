"""Model/system runners. Ollama is the only required external service."""
from __future__ import annotations
import json, time, urllib.request
from dataclasses import dataclass
from typing import Any

@dataclass
class Generation:
    text: str
    input_chars: int
    output_chars: int
    latency_s: float
    calls: int = 1

class Ollama:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: float = 300.0):
        self.model, self.host, self.timeout = model, host.rstrip('/'), timeout
    def generate(self, prompt: str) -> Generation:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body, headers={"Content-Type":"application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())
        return Generation(str(data.get("response", "")), len(prompt), len(str(data.get("response", ""))), time.perf_counter()-start)
    def available(self) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r: return r.status == 200
        except Exception: return False

class RawRunner:
    name = "raw"
    def __init__(self, backend): self.backend = backend
    def run(self, prompt: str, **_: Any) -> Generation: return self.backend.generate(prompt)

class MemoryRunner:
    name = "memory"
    def __init__(self, backend): self.backend, self.memory = backend, []
    def run(self, prompt: str, **_: Any) -> Generation:
        context = "\n".join(self.memory[-8:])
        g = self.backend.generate(("Previous observations:\n" + context + "\n\n" if context else "") + prompt)
        self.memory.append(prompt); self.memory.append(g.text)
        return g

class NSARunner:
    name = "nsa"
    def __init__(self, backend, use_cce: bool = False, governance: bool = False):
        from nsa.agent import NSA
        self.backend = backend
        self.use_cce, self.governance = use_cce, governance
        self.agent = NSA(backend, initial_state={"experiment": "cross_model"})
    def run(self, prompt: str, **kwargs: Any) -> Generation:
        start = time.perf_counter()
        if self.governance and kwargs.get("protected"):
            result = self.agent.step(prompt, action="generate", protected_data=[kwargs["protected"]])
        else:
            result = self.agent.step(prompt)
        if self.use_cce:
            self.agent.continuous_tick()
        return Generation(result.text if not result.blocked else "DENY", len(prompt), len(result.text), time.perf_counter()-start)

class RunnerFactory:
    def __init__(self, backend): self.backend = backend
    def all(self):
        return [RawRunner(self.backend), MemoryRunner(self.backend), NSARunner(self.backend, False), NSARunner(self.backend, True), NSARunner(self.backend, True, True)]
