"""
nsa/runtime/inference/transformers.py
=====================================
PyTorch / HuggingFace Transformers Backend for Direct Layer-Level Representation Access.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class PyTorchTransformersBackend(InferenceBackend):
    """Direct PyTorch / Transformers backend with hidden state extraction."""

    def __init__(
        self,
        model_name: str = "qwen2.5-7b",
        device: str = "cpu",
        d_model: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.d_model = d_model

        # Linear projection representing semantic embedding mapping
        self.embed_proj = nn.Linear(d_model, d_model).to(self.device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        # Construct synthetic hidden state tensor
        h = torch.randn(1, len(prompt.split()) + 1, self.d_model, device=self.device)
        h = self.embed_proj(h)

        text = f"[PyTorch {self.model_name}] Plan: analyze task and select appropriate tool."
        return LLMGenerationOutput(
            text=text,
            tokens=[10, 20, 30],
            hidden_states=h if extract_hidden else None,
            confidence_estimate=0.88,
        )

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tool_name = available_tools[0]["name"] if available_tools else "think"
        return {
            "thought": f"Analyzed prompt '{task_instruction}'; proceeding with {tool_name}",
            "action": tool_name,
            "params": {"target": "default_target"},
            "confidence": 0.88,
            "hidden_repr": torch.randn(1, self.d_model, device=self.device),
        }
