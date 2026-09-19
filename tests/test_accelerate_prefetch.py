from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher


def test_prefetcher_matches_region_prefixes(tmp_path):
    prefetcher = AccelerateDiskPrefetcher(
        tmp_path,
        {"layer.1": ("model.layers.1.",), "layer.2": ("model.layers.2.",)},
    )
    prefetcher._index = {
        "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
        "model.layers.1.mlp.down_proj.weight": "model-00001.safetensors",
        "model.layers.2.mlp.down_proj.weight": "model-00002.safetensors",
    }
    groups = prefetcher._keys_for_region("layer.1")
    assert set(groups) == {"model-00001.safetensors"}
    assert len(groups["model-00001.safetensors"]) == 2
