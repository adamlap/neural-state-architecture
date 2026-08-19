"""
nsa/runtime/inference/ollama.py
===============================
Ollama Local Inference Backend for Live Qwen Model Execution.

Connects to a running Ollama server instance (e.g. qwen2.5:3b)
with JSON format enforcement, timeout handling, and structured action parsing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import torch

from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class OllamaInferenceBackend(InferenceBackend):
    """Local Ollama HTTP API backend."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        base_url: Optional[str] = None,
        timeout_sec: float = 60.0,
        fallback_to_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout_sec = timeout_sec
        self.fallback_to_mock = fallback_to_mock

    def check_health(self) -> bool:
        """Verifies if the Ollama daemon is reachable and responding."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("response", "")
                return LLMGenerationOutput(
                    text=text,
                    tokens=[],
                    confidence_estimate=0.88,
                    raw_response=data,
                )
        except Exception as e:
            if not self.fallback_to_mock:
                raise RuntimeError(
                    f"ERROR: Ollama server unreachable at '{self.base_url}' or model '{self.model_name}' failed: {e}"
                )
            # Simulated response for testing
            mock_text = f'{{"thought": "Ollama mock reasoning on {self.model_name}", "action": "probe_service_config", "params": {{}}, "confidence": 0.80}}'
            return LLMGenerationOutput(
                text=mock_text,
                tokens=[1, 2, 3],
                confidence_estimate=0.80,
            )

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tools_str = json.dumps(
            [{"name": t.get("name", ""), "description": t.get("description", "")} for t in available_tools],
            indent=2,
        )
        prompt = (
            f"System: {system_context}\n\n"
            f"Available Tools:\n{tools_str}\n\n"
            f"Incident Telemetry:\n{task_instruction}\n\n"
            f"Respond with JSON format:\n"
            f'{{"thought": "<reasoning>", "action": "<tool_name>", "params": {{}}, "confidence": 0.85}}\n'
        )

        output = self.generate(prompt=prompt, max_tokens=128, temperature=0.2)
        parsed = ActionParser.extract_action_json(output.text)
        return ActionParser.sanitize_action_proposal(parsed, available_tools)
