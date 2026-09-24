"""Tests for router interceptor and dense sublayer instrumentation."""
import torch
import torch.nn as nn
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyPolicy
from nsa.residency.router_interceptor import MoERouterHook, RoutingPrediction, instrument_dense_sublayers
from nsa.residency.types import NeuralRegion


class DummyRouter(nn.Module):
    def __init__(self, hidden=16, num_experts=4):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):
        return self.gate(x)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(2)])


def test_moe_router_hook():
    router = DummyRouter(hidden=8, num_experts=4)
    # Set weights so expert 1 and expert 3 have highest logits for any positive input
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[1].fill_(2.0)
        router.gate.weight[3].fill_(5.0)

    routed: list[RoutingPrediction] = []
    hook = MoERouterHook(
        layer_index=0,
        gate_module=router,
        num_experts_per_tok=2,
        on_route=routed.append,
    )
    hook.install()

    inp = torch.ones(1, 4, 8)
    _ = router(inp)

    assert len(routed) == 1
    pred = routed[0]
    assert pred.layer_index == 0
    # Expert 3 has highest, then expert 1
    assert pred.selected_regions == ("layer.0.expert.3", "layer.0.expert.1")
    assert len(pred.probabilities) == 2
    assert pred.probabilities[0] > pred.probabilities[1]

    hook.remove()
    _ = router(inp)
    assert len(routed) == 1  # No new route event after removal


def test_dense_sublayer_instrumentation():
    model = DummyModel()
    manager = NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=1024*1024, ram_budget_bytes=2*1024*1024))
    # Register regions
    manager.register([
        NeuralRegion("layer.0.attn", size_bytes=100),
        NeuralRegion("layer.0.mlp", size_bytes=200),
        NeuralRegion("layer.1.attn", size_bytes=100),
        NeuralRegion("layer.1.mlp", size_bytes=200),
    ])

    sublayer_events: list[str] = []
    instrumented = instrument_dense_sublayers(
        model, manager, on_sublayer=sublayer_events.append
    )
    assert instrumented == 4  # 2 attn + 2 mlp

    x = torch.ones(1, 4, 16)
    out = model.model.layers[0](x)
    assert out.shape == x.shape

    # Attn signaled next was layer.0.mlp, then mlp signaled next was layer.1.attn
    assert "layer.0.mlp" in sublayer_events
    assert "layer.1.attn" in sublayer_events
    # Check residency events recorded
    event_actions = [e.action for e in manager.events]
    assert "execute" in event_actions


def test_moe_router_hook_uses_final_token_for_batched_logits():
    class BatchedRouter(nn.Module):
        def forward(self, x):
            # Deliberately return [batch, sequence, experts].
            out = torch.zeros(x.shape[0], x.shape[1], 4)
            out[:, -1, 2] = 7.0
            out[:, -1, 1] = 5.0
            return out

    routed = []
    hook = MoERouterHook(layer_index=2, gate_module=BatchedRouter(), num_experts_per_tok=2, on_route=routed.append)
    hook.install()
    _ = hook.gate_module(torch.ones(2, 3, 8))
    hook.remove()

    assert routed[0].selected_regions == ("layer.2.expert.2", "layer.2.expert.1")
