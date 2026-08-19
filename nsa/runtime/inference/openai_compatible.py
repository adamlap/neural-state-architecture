"""
nsa/runtime/inference/openai_compatible.py
===========================================
OpenAI-Compatible & LM Studio Inference Backend for NSA.

Supports:
  - LM Studio running on Windows Host (port 1234) or WSL
  - Ollama OpenAI-compatible endpoint (/v1)
  - LocalAI / vLLM / Text-Generation-WebUI
  - Automatic WSL -> Windows Host gateway resolution
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import BackendMode, InferenceBackend, LLMGenerationOutput


def discover_windows_host_ip() -> Optional[str]:
    """Resolves the Windows host gateway IP when running inside WSL2."""
    try:
        out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        parts = out.strip().split()
        if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
            return parts[2]
    except Exception:
        pass
    return None


class OpenAICompatibleBackend(InferenceBackend):
    """Universal OpenAI-compatible API backend (LM Studio, vLLM, Ollama v1)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "default",
        mode: BackendMode = BackendMode.LMSTUDIO,
        api_key: str = "not-needed",
        timeout_sec: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.mode = mode
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self._resolved_url = self.base_url

        if self.mode != BackendMode.MOCK:
            self._resolve_connection()

    def _resolve_connection(self) -> None:
        """Tests connection to base_url; if in WSL and localhost fails, falls back to Windows host IP."""
        candidates = [self.base_url]
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            win_ip = discover_windows_host_ip()
            if win_ip:
                alt_url = self.base_url.replace("localhost", win_ip).replace("127.0.0.1", win_ip)
                candidates.append(alt_url)

        connected = False
        last_err = None
        for url in candidates:
            try:
                test_req = urllib.request.Request(
                    f"{url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "NSA-LMStudio-Adapter"},
                )
                with urllib.request.urlopen(test_req, timeout=3.0) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        self._resolved_url = url
                        connected = True
                        # Auto-discover active model name if default
                        if self.model_name == "default" or not self.model_name:
                            models = data.get("data", [])
                            if models:
                                self.model_name = models[0].get("id", "default")
                        break
            except Exception as e:
                last_err = e

        if not connected:
            raise ConnectionError(
                f"[OPENAI/LMSTUDIO BACKEND ERROR] Could not connect to {self.base_url} (or Windows gateway {candidates}). "
                f"Ensure LM Studio / OpenAI server is running and local server is active. Last error: {last_err}"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        if self.mode == BackendMode.MOCK:
            return LLMGenerationOutput(
                text='{"thought": "Mock LMStudio proposal", "action": "probe_service_config", "params": {}}',
                tokens=[101, 102],
                confidence_estimate=0.85,
            )

        endpoint = f"{self._resolved_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")

        return LLMGenerationOutput(
            text=content,
            tokens=[],
            confidence_estimate=0.90,
            raw_response=data,
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

        endpoint = f"{self._resolved_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 160,
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")

        parsed = ActionParser.extract_action_json(content)
        return ActionParser.sanitize_action_proposal(
            parsed, available_tools, default_fallback=fallback_action, strict_live=True
        )


class LMStudioInferenceBackend(OpenAICompatibleBackend):
    """Dedicated alias for LM Studio on Windows Host (port 1234)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_name: str = "default",
        mode: BackendMode = BackendMode.LMSTUDIO,
        timeout_sec: float = 60.0,
    ):
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            mode=mode,
            api_key="lm-studio",
            timeout_sec=timeout_sec,
        )
