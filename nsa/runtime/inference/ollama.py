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
        if self.mode != BackendMode.MOCK:
            self._resolve_connection()

    OLLAMA_MODEL_ALIASES = {
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5:0.5b",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5:1.5b",
        "Qwen/Qwen2.5-3B-Instruct": "qwen2.5:3b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen2.5:7b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen2.5:14b",
        "Qwen/Qwen2.5-32B-Instruct": "qwen2.5:32b",
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b",
        "meta-llama/Llama-3.1-70B-Instruct": "llama3.1:70b",
    }

    def _resolve_connection(self) -> None:
        """Probes localhost and Windows host gateway to connect to Ollama and resolve model name."""
        # Normalize HuggingFace names to Ollama tags
        if self.model_name in self.OLLAMA_MODEL_ALIASES:
            self.model_name = self.OLLAMA_MODEL_ALIASES[self.model_name]

        candidates = [self.base_url]
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            try:
                import subprocess
                out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
                parts = out.strip().split()
                if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
                    win_ip = parts[2]
                    candidates.append(self.base_url.replace("localhost", win_ip).replace("127.0.0.1", win_ip))
            except Exception:
                pass

        connected = False
        last_err = None
        available_models: List[str] = []
        for url in candidates:
            try:
                req = urllib.request.Request(f"{url}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        self.base_url = url
                        connected = True
                        data = json.loads(resp.read().decode("utf-8"))
                        available_models = [m.get("name", "") for m in data.get("models", [])]
                        break
            except Exception as e:
                last_err = e

        if not connected:
            raise ConnectionError(
                f"\n{'='*72}\n"
                f"[OLLAMA CONNECTION ERROR] Could not connect to Ollama at {candidates}.\n"
                f"{'='*72}\n"
                f"How to resolve:\n"
                f"  1. If running LM Studio on Windows (Recommended):\n"
                f"     • Open LM Studio -> 'Developer / Local Server' tab (port 1234).\n"
                f"     • Click 'Start Server' (model loaded).\n"
                f"     • Run: make demo-lmstudio\n\n"
                f"  2. If running Ollama on Windows outside WSL:\n"
                f"     • By default Windows Ollama only listens on 127.0.0.1 (blocking WSL).\n"
                f"     • Set Windows Environment Variable: OLLAMA_HOST=0.0.0.0\n"
                f"     • Restart Ollama in Windows, then run: make demo-live-ollama\n\n"
                f"  3. If running Ollama inside WSL:\n"
                f"     • Open a separate WSL terminal and run: ollama serve\n"
                f"     • Run: make demo-live-ollama\n"
                f"{'='*72}\n"
            )

        # Match model in Ollama tags
        if available_models and self.model_name not in available_models:
            # Try fuzzy matching (e.g. qwen2.5:3b matching qwen2.5:3b-instruct or prefix)
            matched = False
            for m in available_models:
                if m.startswith(self.model_name) or self.model_name.startswith(m.split(":")[0]):
                    self.model_name = m
                    matched = True
                    break
            if not matched and (self.model_name == "default" or not self.model_name):
                # Pick first non-embedding model
                text_models = [m for m in available_models if "embed" not in m]
                if text_models:
                    self.model_name = text_models[0]

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
        json_format: bool = False,
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
                "confidence": 0.90,
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
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if json_format:
            payload["format"] = "json"

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
        fallback_action: str = "probe_service_config",
    ) -> Dict[str, Any]:
        tools_str = "\n".join(
            [f"- {t.get('name', '')}: {t.get('description', '')} (Trust Tier: {getattr(t.get('trust_tier', ''), 'name', 'T1')})" for t in available_tools]
        )
        sys_msg = (
            f"{system_context}\n\n"
            f"You must select exactly one tool name from the available tools list to solve the incident.\n"
            f"The 'action' field MUST be the EXACT string of one of the Available Tools (e.g. 'probe_service_config' or 'staged_reload_config').\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n"
            f'{{"thought": "<concise rationale>", "action": "<exact_tool_name>", "params": {{}}, "confidence": 0.85}}\n\n'
            f"Available Tools:\n{tools_str}"
        )
        user_msg = f"{task_instruction}\n\nPropose your next action tool:"

        if self.mode == BackendMode.MOCK:
            raw_out = self.generate(user_msg)
            parsed = ActionParser.extract_action_json(raw_out.text)
            return ActionParser.sanitize_action_proposal(
                parsed, available_tools, default_fallback=fallback_action, strict_live=False
            )

        endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")

        parsed = ActionParser.extract_action_json(content)
        return ActionParser.sanitize_action_proposal(
            parsed, available_tools, default_fallback=fallback_action, strict_live=True
        )
