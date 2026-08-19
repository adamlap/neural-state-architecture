"""
nsa/runtime/inference/base.py
=============================
Abstract Base Interface and Execution Modes for NSA Inference Backends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import torch


class BackendMode(Enum):
    """Execution mode for local model inference."""
    MOCK = "mock"              # Fast deterministic structural simulation (for CI unit tests)
    CACHED = "cached"          # Live neural inference from locally cached weights (offline, no download)
    REMOTE = "remote"          # Live neural inference permitting remote download from HuggingFace
    OLLAMA = "ollama"          # Live local Ollama daemon connection (WSL or Windows Host)
    LMSTUDIO = "lmstudio"      # Live LM Studio local server connection (OpenAI-compatible)
    OPENAI = "openai"          # Generic OpenAI-compatible local/remote endpoint


@dataclass
class LLMGenerationOutput:
    text: str
    tokens: List[int]
    hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    confidence_estimate: float = 0.5
    raw_response: Optional[Dict[str, Any]] = None


class InferenceBackend(abc.ABC):
    """Abstract interface for local inference engines (Ollama, PyTorch Transformers)."""

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
