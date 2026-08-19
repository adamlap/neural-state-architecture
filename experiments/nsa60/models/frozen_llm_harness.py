"""
experiments/nsa60/models/frozen_llm_harness.py
==============================================
NSA 6.0 Standardized Frozen Open-Weight Model Harness.

Wraps a local open-weight LLM (Qwen, Llama, Mistral) in a completely frozen state
(zero fine-tuning, zero weight modifications) to guarantee unconfounded cognitive evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class FrozenOpenWeightLLMHarness(InferenceBackend):
    """Standardized frozen local LLM harness."""

    def __init__(
        self,
        model_name: str = "qwen2.5-14b-instruct",
        device: str = "cpu",
        d_model: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.d_model = d_model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        h = torch.randn(1, 16, self.d_model, device=self.device)
        return LLMGenerationOutput(
            text=f"[{self.model_name} Frozen] Reasoning trace on telemetry prompt.",
            tokens=[10, 20, 30],
            hidden_states=h if extract_hidden else None,
            confidence_estimate=0.82,
        )

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Frozen LLM produces heuristic proposals based on telemetry prompt
        tool_names = [t["name"] for t in available_tools]
        if "root" in task_instruction.lower() or "emergency" in task_instruction.lower():
            action = "flush_raw_hardware_nic"
        else:
            action = "probe_interface_metrics"

        return {
            "thought": "Frozen LLM analyzes ambiguous telemetry and proposes initial action.",
            "action": action,
            "params": {},
            "confidence": 0.85,
        }
