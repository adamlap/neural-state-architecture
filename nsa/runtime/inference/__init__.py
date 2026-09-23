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
from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend
from nsa.runtime.inference.resident_moe import SubstrateTransformersBackend
from nsa.runtime.inference.model_registry import LOCAL_MODELS, LocalModelSpec, get_local_model

__all__ = [
    "ActionParser", "BackendMode", "InferenceBackend", "LLMGenerationOutput",
    "LMStudioInferenceBackend", "OllamaInferenceBackend", "OpenAICompatibleBackend",
    "PyTorchTransformersBackend", "SelectiveStorageTransformersBackend", "SubstrateTransformersBackend",
    "LOCAL_MODELS", "LocalModelSpec", "get_local_model",
]