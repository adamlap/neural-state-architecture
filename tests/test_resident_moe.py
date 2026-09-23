"""Tests for SubstrateTransformersBackend (MoE and Dense support)."""
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.resident_moe import SubstrateTransformersBackend
from nsa.residency.router_interceptor import RoutingPrediction


def test_substrate_backend_mock_mode():
    backend = SubstrateTransformersBackend(
        model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        mode=BackendMode.MOCK,
    )
    assert backend.mode == BackendMode.MOCK
    loaded = backend.load_model()
    assert loaded is True

    # Generation in mock mode
    out = backend.generate("Hello world", max_tokens=16)
    assert out.text is not None
    assert "thought" in out.text
    assert out.confidence_estimate >= 0.9
    assert "residency" in out.raw_response

    # Action proposal in mock mode
    tools = [{"name": "probe_service_config", "description": "Test tool"}]
    action = backend.propose_action("system", "do something", available_tools=tools)
    assert action["action"] == "probe_service_config"

    backend.close()


def test_moe_routing_prediction_feed():
    backend = SubstrateTransformersBackend(
        model_name="test-moe",
        mode=BackendMode.MOCK,
    )
    # Register dummy regions
    from nsa.residency.types import NeuralRegion
    backend.residency.register([
        NeuralRegion("layer.0.expert.0", size_bytes=1000),
        NeuralRegion("layer.0.expert.1", size_bytes=1000),
        NeuralRegion("layer.0.expert.2", size_bytes=1000),
        NeuralRegion("layer.0.expert.3", size_bytes=1000),
    ])

    # Simulate router prediction event
    pred = RoutingPrediction(
        layer_index=0,
        selected_regions=("layer.0.expert.1", "layer.0.expert.3"),
        probabilities=(0.85, 0.15),
        timestamp=100.0,
    )
    backend._on_moe_route_predicted(pred)

    # When residency plans, expert 1 and 3 should be top scores due to router prediction
    decisions = backend.residency.plan({"tags": []})
    top_decisions = [d.region_id for d in decisions[:2]]
    assert "layer.0.expert.1" in top_decisions

    backend.close()


def test_dense_sublayer_notification():
    backend = SubstrateTransformersBackend(
        model_name="test-dense",
        mode=BackendMode.MOCK,
    )
    mock_controller = MagicMock()
    backend.residency_controller = mock_controller

    backend._on_dense_sublayer_predicted("layer.0.mlp")
    mock_controller.prefetch_async.assert_called_once()

    backend.close()