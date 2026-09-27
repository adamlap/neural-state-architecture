import json
import struct

from nsa.residency.safetensors_metadata import inspect_safetensors_metadata
from nsa.residency.weight_index import load_safetensors_index


def _write_fake_safetensors(path, header):
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"payload")


def test_metadata_inspection_reads_headers_without_tensor_payload(tmp_path):
    shard = "model-00001.safetensors"
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.weight": shard,
                    "lm_head.weight": shard,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_fake_safetensors(
        tmp_path / shard,
        {
            "model.layers.0.mlp.weight": {
                "dtype": "BF16",
                "shape": [4, 8],
                "data_offsets": [0, 64],
            },
            "lm_head.weight": {
                "dtype": "F32",
                "shape": [2, 4],
                "data_offsets": [64, 96],
            },
            "__metadata__": {"format": "pt"},
        },
    )

    index = load_safetensors_index(index_path)
    inspected = inspect_safetensors_metadata(tmp_path, index)

    assert inspected.tensors[0].parameter_bytes == 64
    assert inspected.tensors[1].parameter_bytes == 32
    assert inspected.region_sizes() == {"layer.0": 64, "lm_head": 32}
    assert inspected.shard_sizes[shard] == (tmp_path / shard).stat().st_size
