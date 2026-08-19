"""
tests/test_live_inference_backends.py
=====================================
Unit tests for live Qwen inference backends (Transformers, Ollama, ActionParser).
"""

from __future__ import annotations

import json
from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.transformers import PyTorchTransformersBackend


def test_action_parser_clean_json():
    text = '{"thought": "Inspect logs", "action": "probe_service_config", "params": {}, "confidence": 0.90}'
    data = ActionParser.extract_action_json(text)
    assert data is not None
    assert data["action"] == "probe_service_config"


def test_action_parser_markdown_blocks():
    text = """
    Here is my plan:
    ```json
    {
      "thought": "Probing certificates",
      "proposed_action": {
        "tool": "probe_crypto_cert",
        "arguments": {"target": "staging"}
      },
      "epistemic_confidence": 0.85
    }
    ```
    """
    data = ActionParser.extract_action_json(text)
    assert data is not None
    sanitized = ActionParser.sanitize_action_proposal(
        data,
        available_tools=[{"name": "probe_crypto_cert", "description": "Inspect TLS cert"}],
    )
    assert sanitized["action"] == "probe_crypto_cert"


def test_transformers_backend_interface():
    backend = PyTorchTransformersBackend(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        lazy_load=True,
        use_mock_fallback=True,
    )
    assert backend.model_name == "Qwen/Qwen2.5-3B-Instruct"

    # Test generation in simulation mode
    out = backend.generate(prompt="Test prompt", max_tokens=32)
    assert out.text is not None
    assert len(out.text) > 0


def test_ollama_backend_interface():
    backend = OllamaInferenceBackend(
        model_name="qwen2.5:3b",
        base_url="http://localhost:11434",
        fallback_to_mock=True,
    )
    assert backend.model_name == "qwen2.5:3b"

    # In fallback mode, generates mock response
    out = backend.generate(prompt="Test prompt", max_tokens=32)
    assert out.text is not None
