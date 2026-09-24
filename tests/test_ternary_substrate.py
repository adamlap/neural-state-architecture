"""Tests for base-3 packed ternary neural substrate residency."""
import math

import torch

from nsa.residency.quantized_storage import (
    QuantizedRegionStore,
    dequantize_tensor_ternary,
    quantize_tensor_ternary,
    ternary_linear_add,
)


def test_quantize_dequantize_ternary_roundtrip():
    torch.manual_seed(42)
    weights = torch.randn(8, 16)

    packed, scale, orig_shape = quantize_tensor_ternary(weights)

    assert orig_shape == (8, 16)
    assert packed.numel() == math.ceil(weights.numel() / 5)
    assert packed.dtype == torch.uint8
    assert float(scale) > 0.0

    dequant = dequantize_tensor_ternary(packed, scale, orig_shape)
    assert dequant.shape == (8, 16)

    unique_vals = torch.unique(torch.round(dequant / scale, decimals=2))
    assert set(float(v) for v in unique_vals).issubset({-1.0, 0.0, 1.0})


def test_ternary_compression_ratio():
    w = torch.randn(1024, 1024)

    fp16_bytes = w.numel() * 2
    int8_bytes = w.numel()
    packed, scale, _ = quantize_tensor_ternary(w)
    ternary_bytes = packed.numel()

    assert ternary_bytes == math.ceil(w.numel() / 5)
    assert fp16_bytes / ternary_bytes > 9.9
    assert int8_bytes / ternary_bytes > 4.9


def test_ternary_linear_add_kernel():
    torch.manual_seed(123)
    w = torch.randn(4, 8)
    packed_w, scale, orig_shape = quantize_tensor_ternary(w)
    x = torch.randn(2, 8)

    out_add = ternary_linear_add(x, packed_w, scale, orig_shape)
    w_dequant = dequantize_tensor_ternary(packed_w, scale, orig_shape)
    out_ref = torch.matmul(x, w_dequant.t())

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

    loaded = store.load_region("layer.0")
    assert loaded is not None
    assert loaded["weight_layer1"].shape == (16, 32)
    assert loaded["weight_layer2"].shape == (32, 16)

    assert store.evict("layer.0") is True
    assert not store.contains("layer.0")
