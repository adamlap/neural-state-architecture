from nsa.residency.quantization import (
    QuantizationRegistry,
    QuantizationSpec,
    packed_storage_bytes,
    quantization_spec_from_config,
)


def test_packed_storage_accounts_block_metadata():
    assert packed_storage_bytes(
        (128, 128), 4, block_shape=(128, 128), metadata_bytes_per_block=2
    ) == 8194


def test_registry_describes_storage_and_compute_forms():
    registry = QuantizationRegistry()
    registry.register(
        "layer.0.weight",
        QuantizationSpec("gptq", 4, block_shape=(128, 128), metadata_bytes_per_block=2),
    )
    tensor = registry.describe("layer.0.weight", (128, 128), materialized_bytes=32768)
    assert tensor is not None
    assert tensor.storage_bytes == 8194
    assert tensor.compute_bytes == 32768


def test_hf_fp8_config_normalizes_to_block_quantization():
    spec = quantization_spec_from_config({
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    })
    assert spec is not None
    assert spec.scheme == "fp8"
    assert spec.bits == 8
    assert spec.block_shape == (128, 128)
