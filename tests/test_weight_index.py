from nsa.residency.weight_index import from_tensor_metadata


def test_weight_index_maps_tensors_to_logical_regions():
    index = {
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.3.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.3.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
            "visual.patch_embed.weight": "model-00001-of-00002.safetensors",
        }
    }
    metadata = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [16, 8]},
        "model.layers.3.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8, 8]},
        "model.layers.3.mlp.down_proj.weight": {"dtype": "BF16", "shape": [8, 16]},
        "lm_head.weight": {"dtype": "BF16", "shape": [16, 8]},
        "visual.patch_embed.weight": {"dtype": "F32", "shape": [4, 4]},
    }

    result = from_tensor_metadata(index, metadata)

    assert result.total_bytes == (16 * 8 + 8 * 8 + 8 * 16 + 16 * 8) * 2 + 4 * 4 * 4
    assert result.regions() == (
        "embeddings", "layer.3", "lm_head", "vision"
    )
    assert result.tensors[1].parameter_bytes == 8 * 8 * 2
