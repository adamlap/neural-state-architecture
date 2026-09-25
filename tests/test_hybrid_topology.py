from types import SimpleNamespace

from nsa.residency.hybrid_topology import compile_hybrid_topology


def test_hybrid_topology_compiles_qwen_style_four_layer_cycle():
    config = SimpleNamespace(
        num_hidden_layers=8,
        full_attention_interval=4,
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
    )

    topology = compile_hybrid_topology(config)

    assert len(topology.layers) == 8
    assert [layer.kind for layer in topology.layers] == [
        "linear_attention", "linear_attention", "linear_attention",
        "full_attention", "linear_attention", "linear_attention",
        "linear_attention", "full_attention",
    ]
    assert topology.layers[0].regions[1].persistent is True
    assert topology.layers[3].regions[1].kind == "full_attention"
