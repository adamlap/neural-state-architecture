"""
nsa/runtime/inference/transformers.py
=====================================
PyTorch / HuggingFace Transformers Backend with Strict Mode Separation.

Supports:
- BackendMode.MOCK: Fast deterministic simulation for unit tests.
- BackendMode.CACHED: Strict local offline inference from downloaded checkpoints.
- BackendMode.REMOTE: Live inference permitting download from HuggingFace Hub.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import torch

from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import (
    BackendMode,
    InferenceBackend,
    LLMGenerationOutput,
)


class PyTorchTransformersBackend(InferenceBackend):
    """Direct PyTorch / HuggingFace Transformers inference backend."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        mode: Optional[Union[BackendMode, str]] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        lazy_load: bool = True,
        enable_remote_download: bool = False,
        use_mock_fallback: bool = True,
        d_model: int = 64,
    ) -> None:
        self.model_name = model_name
        if mode is not None:
            self.mode = BackendMode(mode) if isinstance(mode, str) else mode
        elif enable_remote_download:
            self.mode = BackendMode.REMOTE
        elif use_mock_fallback:
            self.mode = BackendMode.MOCK
        else:
            self.mode = BackendMode.CACHED

        self.lazy_load = lazy_load
        self.d_model = d_model

        # Device determination
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Precision selection
        if torch_dtype is None:
            if self.device.type == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        if not lazy_load and self.mode != BackendMode.MOCK:
            self.load_model()

    def load_model(self) -> bool:
        """Loads actual tokenizer and causal language model weights from disk / Hub."""
        if self._is_loaded:
            return True

        if self.mode == BackendMode.MOCK:
            self._is_loaded = True
            return True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            local_only = (self.mode == BackendMode.CACHED)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device.type != "cpu" else None,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self._is_loaded = True
            return True
        except Exception as e:
            self._is_loaded = False
            raise RuntimeError(
                f"ERROR: Failed to load real model '{self.model_name}' in mode '{self.mode.value}': {e}\n"
                f"Hint: Use mode='remote' to download missing weights, or mode='mock' for fast CI tests."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        if self.mode == BackendMode.MOCK:
            h = torch.randn(1, len(prompt.split()) + 1, self.d_model, device=self.device)
            mock_text = json.dumps({
                "thought": f"Mock simulated reasoning for {self.model_name}",
                "action": "probe_service_config",
                "params": {},
                "confidence": 0.88,
            })
            return LLMGenerationOutput(
                text=mock_text,
                tokens=[10, 20, 30],
                hidden_states=h if extract_hidden else None,
                confidence_estimate=0.88,
            )

        if not self._is_loaded:
            self.load_model()

        assert self.model is not None and self.tokenizer is not None

        # Autoregressive generation through genuine neural weights
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0.0,
                "temperature": max(0.01, temperature) if temperature > 0.0 else None,
                "output_hidden_states": extract_hidden,
                "return_dict_in_generate": True,
            }
            outputs = self.model.generate(
                **inputs,
                **{k: v for k, v in gen_kwargs.items() if v is not None}
            )

        gen_tokens = outputs.sequences[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        hidden = outputs.hidden_states[-1][-1] if extract_hidden and outputs.hidden_states else None

        return LLMGenerationOutput(
            text=text,
            tokens=gen_tokens.tolist(),
            hidden_states=hidden,
            confidence_estimate=0.90,
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

        # In live mode (CACHED or REMOTE), format with ChatML template
        if not self._is_loaded:
            self.load_model()

        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        raw_out = self.generate(formatted_prompt, max_tokens=128, temperature=0.0)
        parsed = ActionParser.extract_action_json(raw_out.text)

        # In live mode, strict_live=True raises ValueError if unparseable / invalid tool
        return ActionParser.sanitize_action_proposal(
            parsed, available_tools, default_fallback=fallback_action, strict_live=True
        )
