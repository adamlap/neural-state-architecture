"""
nsa/runtime/inference/base.py
=============================
Abstract Base Interface for NSA Local LLM Inference Backends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class LLMGenerationOutput:
    text: str
    tokens: List[int]
    hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    confidence_estimate: float = 0.5
    raw_response: Optional[Dict[str, Any]] = None


class InferenceBackend(abc.ABC):
    """Abstract interface for local inference engines (Ollama, llama.cpp, PyTorch)."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        """Generate text and extract intermediate neural representations."""
        pass

    @abc.abstractmethod
    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Ask LLM to select an action proposal and provide reasoning."""
        pass
