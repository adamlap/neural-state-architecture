"""Region sizing from checkpoint metadata, with a config-based fallback."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from nsa.residency.accelerate_prefetch import read_safetensors_header

_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")


def decoder_layer_bytes_from_checkpoint(checkpoint_dir: str | Path) -> dict[int, int]:
    """Sum the stored bytes of every ``model.layers.<i>.*`` tensor, per layer.

    Reads only safetensors headers. Returns ``{}`` when the checkpoint is not a
    local safetensors checkpoint.
    """
    root = Path(checkpoint_dir).expanduser()
    if not root.is_dir():
        return {}
    index_file = root / "model.safetensors.index.json"
    file_of: dict[str, str] = {}
    if index_file.is_file():
        try:
            weight_map = json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError):
            weight_map = {}
        file_of = {str(k): str(v) for k, v in weight_map.items()}
    files = sorted(set(file_of.values())) or [p.name for p in sorted(root.glob("*.safetensors"))]
    sizes: dict[int, int] = {}
    for name in files:
        try:
            header = read_safetensors_header(root / name)
        except (OSError, ValueError):
            continue
        for key, (start, end) in header.items():
            match = _LAYER_KEY.match(key)
            if match and (not file_of or file_of.get(key) == name):
                layer = int(match.group(1))
                sizes[layer] = sizes.get(layer, 0) + (end - start)
    return sizes


def estimate_decoder_layer_bytes(config: Any, bytes_per_param: int = 2) -> int:
    """Estimate one Llama/Qwen-style decoder layer from its config.

    Attention: q and o are hidden x hidden, k and v are hidden x (kv_heads * head_dim).
    MLP: gate, up and down projections (3 x hidden x intermediate).
    Plus optional q/k/v biases and the two RMSNorm vectors.
    """
    hidden = int(getattr(config, "hidden_size", 0))
    heads = max(1, int(getattr(config, "num_attention_heads", 1)))
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    inter = int(getattr(config, "intermediate_size", hidden * 4))
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    kv_dim = kv_heads * head_dim
    q_dim = heads * head_dim
    attention = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    biases = q_dim + 2 * kv_dim  # Qwen2 uses qkv bias; slight overestimate elsewhere
    mlp = 3 * hidden * inter
    norms = 2 * hidden
    return int((attention + biases + mlp + norms) * bytes_per_param)


def layer_sizes(config: Any, checkpoint_dir: Optional[str | Path], bytes_per_param: int = 2) -> list[int]:
    """Per-layer byte sizes: measured from the checkpoint when possible."""
    count = int(getattr(config, "num_hidden_layers", 0))
    measured = decoder_layer_bytes_from_checkpoint(checkpoint_dir) if checkpoint_dir else {}
    fallback = estimate_decoder_layer_bytes(config, bytes_per_param)
    return [measured.get(i, fallback) for i in range(count)]
