from nsa.residency.model_execution import compile_execution_graph, compile_residency_regions


def qwen_like_config():
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 8,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"depth": 27},
    }


def test_config_compiles_hybrid_layers_without_weights():
    regions = compile_residency_regions(qwen_like_config())
    ids = {region.region_id for region in regions}
    assert "embeddings" in ids
    assert "layer.3.state" in ids
    assert "layer.0.state" in ids
    assert "lm_head" in ids
    assert "vision" in ids
    assert "mtp" in ids
    assert all(region.size_bytes == 0 for region in regions)


def test_config_compiles_exact_region_sizes_when_index_is_available():
    sizes = {"embeddings": 100, "layer.0.weights": 200, "layer.0.state": 8, "lm_head": 300}
    regions = compile_residency_regions(qwen_like_config(), region_size_bytes=sizes)
    by_id = {region.region_id: region for region in regions}
    assert by_id["embeddings"].size_bytes == 100
    assert by_id["layer.0.weights"].size_bytes == 200
    assert by_id["layer.0.state"].size_bytes == 8
    assert by_id["lm_head"].size_bytes == 300


def test_execution_graph_has_hybrid_state_and_linear_operations():
    graph = compile_execution_graph(qwen_like_config())
    ops = graph.topological_order()
    assert len(ops) == 10
    assert ops[0].op_id == "embed"
    assert ops[1].operation == "linear_attention"
    assert ops[4].operation == "full_attention"
    assert ops[1].persistent_regions == ("layer.0.state",)
    assert ops[4].persistent_regions == ("layer.3.state",)
    assert ops[-1].op_id == "lm_head"
