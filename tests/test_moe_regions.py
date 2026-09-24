"""Tests for architecture-derived MoE residency paths."""
from types import SimpleNamespace

import torch.nn as nn

from nsa.residency.moe_regions import build_moe_regions, infer_moe_spec_from_model


class DummyExperts(nn.Module):
    def __init__(self, count=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(count)])


class DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4)
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = nn.Linear(8, 8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = DummyMLP()


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer()])


def config():
    return SimpleNamespace(
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        hidden_size=8,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
        model_type="future_moe_architecture",
    )


def test_infer_moe_spec_uses_actual_module_tree():
    spec = infer_moe_spec_from_model(config(), DummyModel())

    assert spec is not None
    assert spec.expert_container == "mlp.experts"
    assert spec.router_path == "mlp.gate"
    assert "mlp\\.experts" in spec.expert_pattern


def test_build_moe_regions_uses_inferred_paths():
    spec = infer_moe_spec_from_model(config(), DummyModel())
    regions = build_moe_regions(config(), spec=spec)

    expert_regions = [r for r in regions if ".expert." in r.region_id]
    assert len(expert_regions) == 4
    assert all("model.layers.0.mlp.experts." in r.parameter_prefixes[0] for r in expert_regions)
    shared = next(r for r in regions if r.region_id == "layer.0.shared")
    assert "model.layers.0.mlp.gate." in shared.parameter_prefixes
