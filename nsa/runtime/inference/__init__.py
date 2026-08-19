"""
nsa/runtime/inference
=====================
NSA Inference Backends (Transformers, Ollama, LM Studio / OpenAI-Compatible).
"""

from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import BackendMode, InferenceBackend, LLMGenerationOutput
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.openai_compatible import (
    LMStudioInferenceBackend,
    OpenAICompatibleBackend,
    discover_windows_host_ip,
)
from nsa.runtime.inference.transformers import PyTorchTransformersBackend

__all__ = [
    "ActionParser",
    "BackendMode",
    "InferenceBackend",
    "LLMGenerationOutput",
    "LMStudioInferenceBackend",
    "OllamaInferenceBackend",
    "OpenAICompatibleBackend",
    "PyTorchTransformersBackend",
    "discover_windows_host_ip",
]
