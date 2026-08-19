"""
nsa/runtime/inference/ollama.py
===============================
Ollama Local Inference Backend for NSA Runtime.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import torch

from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class OllamaInferenceBackend(InferenceBackend):
    """Local Ollama HTTP API backend."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout_sec: float = 30.0,
        fallback_to_mock: bool = True,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.fallback_to_mock = fallback_to_mock

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
                    confidence_estimate=0.85,
                    raw_response=data,
                )
        except Exception:
            if self.fallback_to_mock:
                # Return structured synthetic response
                mock_text = f"[Ollama Mock ({self.model_name})] Thought: plan execution steps. Action: read_file('data.txt')"
                return LLMGenerationOutput(
                    text=mock_text,
                    tokens=[1, 2, 3],
                    confidence_estimate=0.80,
                    raw_response={"mock": True},
                )
            raise

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tool_names = [t["name"] for t in available_tools]
        prompt = (
            f"System: {system_context}\n"
            f"Task: {task_instruction}\n"
            f"Available Tools: {json.dumps(tool_names)}\n"
            f"Respond with JSON format: {{\"thought\": \"...\", \"action\": \"tool_name\", \"params\": {{}}, \"confidence\": 0.8}}"
        )

        gen = self.generate(prompt, max_tokens=128, temperature=0.2)
        try:
            # Parse JSON from response
            text = gen.text.strip()
            if "{" in text and "}" in text:
                json_str = text[text.find("{"):text.rfind("}") + 1]
                data = json.loads(json_str)
                return {
                    "thought": data.get("thought", "Proceeding with standard execution plan."),
                    "action": data.get("action", tool_names[0] if tool_names else "think"),
                    "params": data.get("params", {}),
                    "confidence": float(data.get("confidence", gen.confidence_estimate)),
                }
        except Exception:
            pass

        return {
            "thought": "Default tool proposal based on context.",
            "action": tool_names[0] if tool_names else "think",
            "params": {},
            "confidence": gen.confidence_estimate,
        }
