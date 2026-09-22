import json
import struct
from types import SimpleNamespace

from nsa.residency.sizing import decoder_layer_bytes_from_checkpoint, estimate_decoder_layer_bytes, layer_sizes


def _write(path, tensors):
    header, cursor = {}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [cursor, cursor + size]}
        cursor += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * cursor)


def test_measured_sizes_sum_per_layer_across_shards(tmp_path):
    _write(tmp_path / "a.safetensors", {"model.layers.0.w": 10, "model.layers.1.w": 20, "model.embed_tokens.weight": 999})
    _write(tmp_path / "b.safetensors", {"model.layers.0.b": 5})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.0.w": "a.safetensors", "model.layers.1.w": "a.safetensors",
        "model.embed_tokens.weight": "a.safetensors", "model.layers.0.b": "b.safetensors"}}))
    assert decoder_layer_bytes_from_checkpoint(tmp_path) == {0: 15, 1: 20}


def test_single_file_checkpoint_and_non_local_path(tmp_path):
    _write(tmp_path / "model.safetensors", {"model.layers.3.w": 7})
    assert decoder_layer_bytes_from_checkpoint(tmp_path) == {3: 7}
    assert decoder_layer_bytes_from_checkpoint("Qwen/Qwen2.5-3B-Instruct") == {}


def test_config_estimate_matches_real_qwen_layer_size():
    # Qwen2.5-3B: 36 layers, ~77M parameters/layer -> ~147 MiB in bf16 (was overestimated ~39%)
    cfg = SimpleNamespace(hidden_size=2048, intermediate_size=11008, num_attention_heads=16, num_key_value_heads=2, num_hidden_layers=36)
    mib = estimate_decoder_layer_bytes(cfg) / 2**20
    assert 145 <= mib <= 150
    cfg15 = SimpleNamespace(hidden_size=1536, intermediate_size=8960, num_attention_heads=12, num_key_value_heads=2, num_hidden_layers=28)
    assert 88 <= estimate_decoder_layer_bytes(cfg15) / 2**20 <= 91


def test_layer_sizes_prefers_measurement_and_falls_back_per_layer(tmp_path):
    _write(tmp_path / "model.safetensors", {"model.layers.0.w": 111})
    cfg = SimpleNamespace(hidden_size=8, intermediate_size=16, num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=2)
    sizes = layer_sizes(cfg, tmp_path)
    assert sizes[0] == 111 and sizes[1] == estimate_decoder_layer_bytes(cfg)
