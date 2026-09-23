"""Tests for adaptive precision and quantized neural storage."""
import torch
from nsa.residency.quantized_storage import (
    QuantizedRegionStore,
    dequantize_tensor_int4,
    dequantize_tensor_int8,
    quantize_tensor_int4,
    quantize_tensor_int8,
)


def test_quantize_dequantize_int8():
    orig = torch.randn(32, 64, dtype=torch.float32)
    int8_data, scale = quantize_tensor_int8(orig)

    assert int8_data.dtype == torch.int8
    assert int8_data.shape == orig.shape
    assert scale.dtype == torch.float32

    recon = dequantize_tensor_int8(int8_data, scale, dtype=torch.float32)
    # Check max absolute error is bounded by quantization step (scale)
    max_err = torch.max(torch.abs(orig - recon)).item()
    assert max_err <= scale.item() * 1.5


def test_quantize_dequantize_int4():
    orig = torch.randn(16, 32, dtype=torch.float32)
    packed, scale, orig_shape = quantize_tensor_int4(orig)

    assert packed.dtype == torch.uint8
    # 16 * 32 = 512 elements; packed in 4-bit nibbles -> 256 bytes
    assert packed.numel() == 256
    assert orig_shape == (16, 32)

    recon = dequantize_tensor_int4(packed, scale, orig_shape, dtype=torch.float32)
    assert recon.shape == orig.shape
    max_err = torch.max(torch.abs(orig - recon)).item()
    # 4-bit quantization step is scale
    assert max_err <= scale.item() * 2.0


def test_quantized_region_store_lifecycle():
    store_int8 = QuantizedRegionStore(target_precision="int8")
    store_int4 = QuantizedRegionStore(target_precision="int4")

    state_dict = {
        "weight_a": torch.randn(64, 64, dtype=torch.float32),
        "bias_a": torch.randn(64, dtype=torch.float32),
    }
    raw_bytes = sum(t.element_size() * t.numel() for t in state_dict.values())

    int8_bytes = store_int8.store_region("layer.0.expert.1", state_dict)
    int4_bytes = store_int4.store_region("layer.0.expert.1", state_dict)

    # Int8 should be ~25-30% of float32 size; Int4 should be ~15-20%
    assert int8_bytes < raw_bytes * 0.5
    assert int4_bytes < int8_bytes

    assert store_int8.contains("layer.0.expert.1")
    loaded = store_int8.load_region("layer.0.expert.1")
    assert loaded is not None
    assert "weight_a" in loaded and "bias_a" in loaded
    assert loaded["weight_a"].shape == (64, 64)

    assert store_int8.evict("layer.0.expert.1") is True
    assert store_int8.contains("layer.0.expert.1") is False