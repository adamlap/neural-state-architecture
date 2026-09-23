"""Tests for 1.58-bit ternary neural substrate residency (Bonsai-inspired)."""
import torch
import pytest
from nsa.residency.quantized_storage import (
    quantize_tensor_ternary,
    dequantize_tensor_ternary,
    ternary_linear_add,
    QuantizedRegionStore,
)


def test_quantize_dequantize_ternary_roundtrip():
    # Test random tensor
    torch.manual_seed(42)
    weights = torch.randn(8, 16)

    packed, scale, orig_shape = quantize_tensor_ternary(weights)

    # 8 * 16 = 128 elements; 4 trits per byte -> 32 bytes packed
    assert orig_shape == (8, 16)
    assert packed.numel() == 32
    assert packed.dtype == torch.uint8
    assert scale > 0.0

    # Dequantize
    dequant = dequantize_tensor_ternary(packed, scale, orig_shape)
    assert dequant.shape == (8, 16)

    # Values in dequantized tensor must be strictly in {-scale, 0, scale}
    unique_vals = torch.unique(torch.round(dequant / scale, decimals=2))
    for val in unique_vals:
        assert float(val) in (-1.0, 0.0, 1.0)


def test_ternary_compression_ratio():
    # 1024 x 1024 weight matrix (1,048,576 elements)
    w = torch.randn(1024, 1024)

    fp16_bytes = w.numel() * 2  # 2,097,152 bytes (~2 MB)
    int8_bytes = w.numel() * 1  # 1,048,576 bytes (~1 MB)

    packed, scale, _ = quantize_tensor_ternary(w)
    ternary_bytes = packed.numel()  # 262,144 bytes (~256 KB)

    # Exactly 8x compression over FP16, 4x compression over INT8
    assert fp16_bytes / ternary_bytes == 8.0
    assert int8_bytes / ternary_bytes == 4.0


def test_ternary_linear_add_kernel():
    torch.manual_seed(123)
    # Weight: (4 out_features, 8 in_features)
    w = torch.randn(4, 8)
    packed_w, scale, orig_shape = quantize_tensor_ternary(w)

    x = torch.randn(2, 8)  # batch_size=2

    # 1. Addition-only linear execution
    out_add = ternary_linear_add(x, packed_w, scale, orig_shape)

    # 2. Standard matmul using dequantized weights
    w_dequant = dequantize_tensor_ternary(packed_w, scale, orig_shape)
    out_ref = torch.matmul(x, w_dequant.t())

    # Must be mathematically identical
    torch.testing.assert_close(out_add, out_ref, atol=1e-5, rtol=1e-5)


def test_quantized_region_store_ternary_lifecycle():
    store = QuantizedRegionStore(target_precision="ternary")

    state_dict = {
        "weight_layer1": torch.randn(16, 32),
        "weight_layer2": torch.randn(32, 16),
    }

    compressed_bytes = store.store_region("layer.0", state_dict)
    assert compressed_bytes > 0
    assert store.contains("layer.0")

    # Load and verify dequantized tensors
    loaded = store.load_region("layer.0")
    assert loaded is not None
    assert "weight_layer1" in loaded
    assert loaded["weight_layer1"].shape == (16, 32)
    assert loaded["weight_layer2"].shape == (32, 16)

    # Evict
    assert store.evict("layer.0") is True
    assert store.contains("layer.0") is False
