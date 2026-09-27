from nsa.residency.model_plan import build_model_residency_plan
from nsa.residency.weight_index import from_tensor_metadata


def test_model_plan_combines_index_and_execution_graph():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
    }
    index = from_tensor_metadata(
        {
            "weight_map": {
                "model.embed_tokens.weight": "a",
                "model.layers.0.mlp.weight": "a",
                "model.layers.1.mlp.weight": "b",
                "model.layers.2.mlp.weight": "b",
                "model.layers.3.mlp.weight": "c",
                "lm_head.weight": "c",
            }
        },
        {
            "model.embed_tokens.weight": {"dtype": "BF16", "shape": [10, 4]},
            "model.layers.0.mlp.weight": {"dtype": "BF16", "shape": [4, 4]},
            "model.layers.1.mlp.weight": {"dtype": "BF16", "shape": [4, 4]},
            "model.layers.2.mlp.weight": {"dtype": "BF16", "shape": [4, 4]},
            "model.layers.3.mlp.weight": {"dtype": "BF16", "shape": [4, 4]},
            "lm_head.weight": {"dtype": "BF16", "shape": [10, 4]},
        },
    )
    plan = build_model_residency_plan(config, weight_index=index)
    by_id = {r.region_id: r for r in plan.regions}
    assert by_id["layer.0.weights"].size_bytes == 32
    assert by_id["layer.3.weights"].size_bytes == 32
    assert by_id["lm_head"].size_bytes == 80
    assert plan.storage_bytes == index.total_bytes
    assert plan.region_count == 10
    assert len(plan.execution_graph.operations) == 6
