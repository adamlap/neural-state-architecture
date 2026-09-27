from nsa.residency.model_analysis import analyze_plan, qwen3_5_residency_summary
from nsa.residency.model_plan import build_model_residency_plan


def test_qwen35_architecture_compiles_without_weights():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"depth": 27},
    }
    plan = build_model_residency_plan(config)
    analysis = analyze_plan(plan, residency_budget_bytes=1024 * 1024)
    summary = qwen3_5_residency_summary(config, plan)

    assert len(plan.execution_graph.operations) == 66
    assert summary["layers"] == 64
    assert summary["linear_attention_layers"] == 48
    assert summary["full_attention_layers"] == 16
    assert summary["has_vision"] is True
    assert summary["has_mtp"] is True
    assert analysis.operation_count == 66
    assert analysis.peak_required_bytes == 0
