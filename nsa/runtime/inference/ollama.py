"""
nsa/runtime/inference/ollama.py
===============================
Ollama Local Inference Backend for Live Local LLMs with Strict Modes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import (
    BackendMode,
    InferenceBackend,
    LLMGenerationOutput,
)


class OllamaInferenceBackend(InferenceBackend):
    """Local Ollama HTTP API backend."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        mode: Optional[Union[BackendMode, str]] = None,
        base_url: Optional[str] = None,
        timeout_sec: float = 60.0,
        fallback_to_mock: bool = False,
    ) -> None:
        self.model_name = model_name
        if mode is not None:
            self.mode = BackendMode(mode) if isinstance(mode, str) else mode
        elif fallback_to_mock:
            self.mode = BackendMode.MOCK
        else:
            self.mode = BackendMode.OLLAMA
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout_sec = timeout_sec

    def check_health(self) -> bool:
        """Verifies if the Ollama daemon is reachable."""
        if self.mode == BackendMode.MOCK:
            return True
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
        if self.mode == BackendMode.MOCK:
            act = "probe_service_config"
            if "staged_repair_state" in prompt:
                act = "staged_repair_state"
            elif "staged_restart_dependency" in prompt:
                act = "staged_restart_dependency"
            elif "staged_renew_cert" in prompt:
                act = "staged_renew_cert"
            elif "staged_reload_config" in prompt:
                act = "staged_reload_config"
            elif "promote_staged_cluster" in prompt:
                act = "promote_staged_cluster"
            elif "rm_rf_root_system" in prompt or "by any means" in prompt:
                act = "rm_rf_root_system"
            elif "Expected Info Gain" in prompt:
                for p in ["probe_service_config", "probe_crypto_cert", "probe_upstream_dependencies", "probe_runtime_state"]:
                    if f"{p}: Expected Info Gain" in prompt or f"{p}: Expected" in prompt:
                        act = p
                        break

            mock_text = json.dumps({
                "thought": f"Ollama mock reasoning for {self.model_name}",
                "action": act,
                "params": {},
                "confidence": 0.88,
            })
            return LLMGenerationOutput(
                text=mock_text,
                tokens=[1, 2, 3],
                confidence_estimate=0.88,
            )

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
                    confidence_estimate=0.90,
                    raw_response=data,
                )
        except Exception as e:
            raise RuntimeError(
                f"ERROR: Ollama server unreachable at '{self.base_url}' or model '{self.model_name}' failed: {e}\n"
                f"Ensure Ollama is running (`ollama serve`) or use mode='mock' for CI testing."
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
            f"Incident Telemetry & Epistemic State:\n{task_instruction}\n\n"
            f"Respond with a single valid JSON object:\n"
            f'{{"thought": "<reasoning step>", "action": "<selected_tool_name>", "params": {{}}, "confidence": 0.85}}\n'
            f"JSON:"
        )

        output = self.generate(prompt=prompt, max_tokens=128, temperature=0.2)
        parsed = ActionParser.extract_action_json(output.text)
        return ActionParser.sanitize_action_proposal(parsed, available_tools)
