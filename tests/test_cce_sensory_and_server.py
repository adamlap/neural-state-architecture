"""Unit tests for CCE Sensory Ingress and Proxy Server CCE integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from nsa.runtime.cce_sensory import CCESensoryIngress
from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.server.proxy import NSAProxyRuntime


def test_cce_sensory_ingress_encoding():
    sensory = CCESensoryIngress(dimension=4, scale=0.5)
    p1, e1 = sensory.encode_text_to_perturbation("Alert: High CPU utilization", source="sys_monitor", importance=0.9)
    assert p1.shape == (4,)
    assert torch.isfinite(p1).all()
    assert e1.source == "sys_monitor"
    assert e1.importance == 0.9

    p2, e2 = sensory.encode_text_to_perturbation("Normal status", source="sys_monitor", importance=0.1)
    # Different content produces distinct perturbations
    assert not torch.allclose(p1, p2)
    assert len(sensory.recent_events) == 2


def test_proxy_runtime_cce_initialization_and_chat(monkeypatch):
    # Mock Ollama backend to test CCE state integration without live Ollama
    with patch("nsa.server.proxy.OllamaInferenceBackend") as mock_backend_cls:
        mock_backend = MagicMock()
        mock_backend.model_name = "qwen2.5:3b"
        mock_backend.generate.return_value = LLMGenerationOutput(
            text="I acknowledge the continuous sensory input.",
            tokens=[1, 2, 3],
            confidence_estimate=0.95,
        )
        mock_backend.generate_text.return_value = "I acknowledge the continuous sensory input."
        mock_backend_cls.return_value = mock_backend

        runtime = NSAProxyRuntime(backend_type="ollama", model="qwen2.5:3b", enable_cce=True)
        assert runtime.enable_cce is True
        assert runtime.cce_state is not None

        # Test sensor API
        res = runtime.process_sensor_input("Sensor reading: ambient temp 22C", source="temp_sensor", importance=0.4)
        assert res["status"] == "ingested"
        assert "cce_snapshot" in res

        # Test Chat API
        messages = [{"role": "user", "content": "What is your current cognitive status?"}]
        chat_res = runtime.process_chat(messages)

        assert "content" in chat_res
        assert "Continuous Cognitive Engine" in chat_res["content"]
        assert "Wall-Clock Elapsed" in chat_res["content"]

        # Clean up background thread
        runtime._stop_cce.set()
