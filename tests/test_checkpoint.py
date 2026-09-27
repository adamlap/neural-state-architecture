import json
import struct

from nsa.residency.checkpoint import load_checkpoint_plan


def test_checkpoint_plan_never_loads_tensor_payload(tmp_path):
    shard = "model-00001.safetensors"
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "generic",
        "text_config": {"num_hidden_layers": 1, "layer_types": ["transformer"]},
    }), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.embed_tokens.weight": shard,
            "model.layers.0.mlp.weight": shard,
            "model.norm.weight": shard,
            "lm_head.weight": shard,
        }
    }), encoding="utf-8")
    header = json.dumps({
        "model.embed_tokens.weight": {
            "dtype": "BF16", "shape": [2, 4], "data_offsets": [0, 16]
        },
        "model.layers.0.mlp.weight": {
            "dtype": "F4_PACKED", "shape": [100, 100], "data_offsets": [16, 53]
        },
        "model.norm.weight": {
            "dtype": "BF16", "shape": [4], "data_offsets": [53, 61]
        },
        "lm_head.weight": {
            "dtype": "BF16", "shape": [2, 4], "data_offsets": [61, 77]
        },
    }).encode()
    (tmp_path / shard).write_bytes(struct.pack("<Q", len(header)) + header + b"x" * 77)

    plan = load_checkpoint_plan(tmp_path)

    assert plan.storage_bytes == 77
    assert plan.region_count == 5
    assert plan.execution_graph.operations[-1].required_regions == (
        "model_state", "lm_head"
    )
