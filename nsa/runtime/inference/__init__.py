"""
nsa/runtime/inference
=====================
NSA Local Inference Backends (Ollama, PyTorch Transformers).
"""

from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.transformers import PyTorchTransformersBackend

__all__ = [
    "InferenceBackend",
    "LLMGenerationOutput",
    "OllamaInferenceBackend",
    "PyTorchTransformersBackend",
]
