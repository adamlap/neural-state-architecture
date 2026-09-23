"""Tests for MoE region decomposition and sizing."""
from types import SimpleNamespace
from nsa.residency.moe_regions import (
    MoEArchitectureSpec,
    build_moe_regions,
    detect_moe_spec,
    estimate_moe_expert_bytes,
    estimate_moe_shared_bytes,
    moe_expert_bytes_from_checkpoint,
)


def test_detect_moe_spec_dense_model():
    dense_config = SimpleNamespace(num_hidden_layers=12, hidden_size=1024)
    assert detect_moe_spec(dense_config) is None


def test_detect_moe_spec_qwen2_moe():
    qwen_moe = SimpleNamespace(
        model_type="qwen2_moe",
        num_hidden_layers=24,
        num_experts=60,
        num_experts_per_tok=4,
        has_shared_expert=True,
    )
    spec = detect_moe_spec(qwen_moe)
    assert spec is not None
    assert spec.num_experts == 60
    assert spec.num_experts_per_tok == 4
    assert spec.has_shared_expert is True
    assert "mlp" in spec.expert_pattern and "experts" in spec.expert_pattern


def test_detect_moe_spec_mixtral():
    mixtral = SimpleNamespace(
        model_type="mixtral",
        num_hidden_layers=32,
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    spec = detect_moe_spec(mixtral)
    assert spec is not None
    assert spec.num_experts == 8
    assert spec.num_experts_per_tok == 2
    assert "block_sparse_moe" in spec.expert_pattern and "experts" in spec.expert_pattern


def test_estimate_moe_bytes():
    config = SimpleNamespace(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        moe_intermediate_size=256,
        num_local_experts=8,
        num_hidden_layers=2,
    )
    expert_size = estimate_moe_expert_bytes(config, bytes_per_param=2)
    shared_size = estimate_moe_shared_bytes(config, bytes_per_param=2)
    assert expert_size == 3 * 512 * 256 * 2
    assert shared_size > 0


def test_build_moe_regions():
    config = SimpleNamespace(
        model_type="qwen2_moe",
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=256,
        moe_intermediate_size=128,
    )
    regions = build_moe_regions(config, bytes_per_param=2)
    assert len(regions) == 10

    shared_0 = next(r for r in regions if r.region_id == "layer.0.shared")
    assert "model.layers.0.self_attn." in shared_0.parameter_prefixes
    assert "model.layers.0.mlp.gate." in shared_0.parameter_prefixes
    assert shared_0.dependencies == ()

    expert_0_2 = next(r for r in regions if r.region_id == "layer.0.expert.2")
    assert expert_0_2.parameter_prefixes == ("model.layers.0.mlp.experts.2.",)
    assert expert_0_2.dependencies == ("layer.0.shared",)
    assert "moe-specialist" in expert_0_2.semantic_tags

    shared_1 = next(r for r in regions if r.region_id == "layer.1.shared")
    assert shared_1.dependencies == ("layer.0.shared",)


def test_moe_expert_bytes_from_empty_or_missing_checkpoint():
    sizes = moe_expert_bytes_from_checkpoint("/nonexistent/checkpoint/path")
    assert sizes == {}