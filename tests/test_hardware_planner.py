from nsa.residency.hardware_planner import HardwareProfile, plan_for_hardware
from nsa.residency.model_plan import build_model_residency_plan


def test_hardware_plan_is_independent_of_model_architecture():
    plan = build_model_residency_plan(
        {"text_config": {"num_hidden_layers": 2, "layer_types": ["transformer", "transformer"]}},
        region_size_bytes={
            "embeddings": 10,
            "layer.0": 20,
            "layer.1": 20,
            "model_state": 2,
            "lm_head": 10,
        },
    )
    hardware = HardwareProfile(
        "tiny",
        "cpu",
        memory_bytes=25,
        capabilities=frozenset({"matmul"}),
        preferred_quantization=("int4", "fp8"),
    )

    result = plan_for_hardware(plan, hardware)

    assert result.requires_streaming is True
    assert result.supported_quantization == "int4"
    assert result.analysis.fits_peak_budget is False
