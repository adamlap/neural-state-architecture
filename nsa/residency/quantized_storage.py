"""Adaptive precision and quantized storage for neural regions.

Supports storing cold/warm weights in INT8, INT4, or base-3 packed ternary
representations on disk or host RAM, and dynamically dequantizing them into
compute-ready floating point tensors.

Applies equally to:
- Dense transformer layers/sublayers (reducing NVMe-to-RAM I/O bandwidth).
- MoE specialist experts (minimizing fast memory footprint).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch


def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor INT8 quantization."""
    tensor = tensor.contiguous()
    max_val = torch.max(torch.abs(tensor)).clamp(min=1e-8)
    scale = max_val / 127.0
    int8_data = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8_data, scale.to(torch.float32)


def dequantize_tensor_int8(
    int8_data: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Dequantize symmetric INT8 tensor back to float."""
    target_device = device or int8_data.device
    return (int8_data.to(target_device, dtype=dtype) * scale.to(target_device, dtype=dtype)).to(dtype)


def quantize_tensor_int4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Symmetric INT4 quantization packed into uint8 bytes (2 nibbles per byte)."""
    orig_shape = tuple(tensor.shape)
    flat = tensor.contiguous().view(-1)
    pad_len = (2 - (flat.numel() % 2)) % 2
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    max_val = torch.max(torch.abs(flat)).clamp(min=1e-8)
    scale = max_val / 7.0
    int4_vals = torch.clamp(torch.round(flat / scale), -7, 7).to(torch.int8)
    unsigned_int4 = (int4_vals + 7).to(torch.uint8)

    even_vals = unsigned_int4[0::2]
    odd_vals = unsigned_int4[1::2]
    packed = (even_vals << 4) | (odd_vals & 0x0F)

    return packed, scale.to(torch.float32), orig_shape


def dequantize_tensor_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Unpack and dequantize INT4 packed bytes back to floating point tensor."""
    target_device = device or packed.device
    packed_dev = packed.to(target_device)
    even = (packed_dev >> 4) & 0x0F
    odd = packed_dev & 0x0F

    interleaved = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=target_device)
    interleaved[0::2] = even
    interleaved[1::2] = odd

    num_orig = 1
    for s in orig_shape:
        num_orig *= s
    trimmed = interleaved[:num_orig]
    signed = (trimmed.to(dtype=dtype) - 7.0) * scale.to(target_device, dtype=dtype)
    return signed.view(orig_shape).to(dtype)


def quantize_tensor_ternary(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Ternary {-1, 0, +1} quantization using exact base-3 packing.

    Five ternary values (trits) fit in one uint8 because 3**5 == 243.
    This uses 1.6 stored bits/trit, approaching the 1.585-bit information
    limit without wasting the unused states of a 2-bit-per-trit encoding.
    """
    orig_shape = tuple(tensor.shape)
    flat = tensor.contiguous().view(-1).to(torch.float32)
    scale = torch.mean(torch.abs(flat)).clamp(min=1e-8)
    ternary = torch.clamp(torch.round(flat / scale), -1, 1).to(torch.int8)
    encoded = torch.where(
        ternary == 1,
        torch.ones((), dtype=torch.uint8, device=tensor.device),
        torch.where(
            ternary == -1,
            torch.full((), 2, dtype=torch.uint8, device=tensor.device),
            torch.zeros((), dtype=torch.uint8, device=tensor.device),
        ),
    )

    pad_len = (5 - (encoded.numel() % 5)) % 5
    if pad_len > 0:
        encoded = torch.nn.functional.pad(encoded, (0, pad_len))

    t0, t1, t2, t3, t4 = (encoded[i::5] for i in range(5))
    packed = t0 + 3 * t1 + 9 * t2 + 27 * t3 + 81 * t4
    return packed.to(torch.uint8), scale.to(torch.float32), orig_shape


def dequantize_tensor_ternary(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Unpack exact base-3 ternary bytes and dequantize to floating point."""
    target_device = device or packed.device
    packed_dev = packed.to(target_device).to(torch.int64)

    digits = []
    value = packed_dev
    for _ in range(5):
        digits.append(value.remainder(3).to(torch.uint8))
        value = torch.div(value, 3, rounding_mode="floor")

    interleaved = torch.empty(packed.numel() * 5, dtype=torch.uint8, device=target_device)
    for i, digit in enumerate(digits):
        interleaved[i::5] = digit

    num_orig = 1
    for s in orig_shape:
        num_orig *= s
    trimmed = interleaved[:num_orig]
    ternary_f = torch.where(
        trimmed == 1,
        torch.ones((), dtype=dtype, device=target_device),
        torch.where(
            trimmed == 2,
            torch.full((), -1.0, dtype=dtype, device=target_device),
            torch.zeros((), dtype=dtype, device=target_device),
        ),
    )
    return (ternary_f * scale.to(target_device, dtype=dtype)).view(orig_shape).to(dtype)


def ternary_linear_add(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    scale: torch.Tensor,
    w_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Multiplication-free linear forward pass using ternary additions/subtractions."""
    target_device = x.device
    packed_dev = packed_w.to(target_device).to(torch.int64)

    digits = []
    value = packed_dev
    for _ in range(5):
        digits.append(value.remainder(3).to(torch.uint8))
        value = torch.div(value, 3, rounding_mode="floor")

    interleaved = torch.empty(packed_w.numel() * 5, dtype=torch.uint8, device=target_device)
    for i, digit in enumerate(digits):
        interleaved[i::5] = digit

    num_w = w_shape[0] * w_shape[1]
    trimmed = interleaved[:num_w].view(w_shape)
    pos_mask = (trimmed == 1).to(x.dtype)
    neg_mask = (trimmed == 2).to(x.dtype)

    pos_contrib = torch.matmul(x, pos_mask.t())
    neg_contrib = torch.matmul(x, neg_mask.t())
    return (pos_contrib - neg_contrib) * scale.to(target_device, dtype=x.dtype)


@dataclass
class QuantizedWeightBuffer:
    """Holds a quantized tensor buffer with its decompression metadata."""
    data: torch.Tensor
    scale: torch.Tensor
    precision: str
    orig_shape: Tuple[int, ...]
    orig_dtype: torch.dtype

    @property
    def num_bytes(self) -> int:
        return self.data.element_size() * self.data.numel() + self.scale.element_size() * self.scale.numel()

    def dequantize(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        if self.precision == "int8":
            return dequantize_tensor_int8(self.data, self.scale, dtype=self.orig_dtype, device=device)
        if self.precision == "int4":
            return dequantize_tensor_int4(self.data, self.scale, self.orig_shape, dtype=self.orig_dtype, device=device)
        if self.precision == "ternary":
            return dequantize_tensor_ternary(self.data, self.scale, self.orig_shape, dtype=self.orig_dtype, device=device)
        return self.data.to(device=device, dtype=self.orig_dtype)


class QuantizedRegionStore:
    """In-memory or tiered storage for quantized neural region parameter buffers."""

    def __init__(self, target_precision: str = "int8") -> None:
        self.target_precision = target_precision
        self._buffers: Dict[str, Dict[str, QuantizedWeightBuffer]] = {}

    def store_region(self, region_id: str, state_dict: Dict[str, torch.Tensor]) -> int:
        """Quantize and store all parameters for a region. Returns total compressed bytes."""
        region_map: Dict[str, QuantizedWeightBuffer] = {}
        total_bytes = 0

        for param_name, tensor in state_dict.items():
            if self.target_precision == "int8":
                data, scale = quantize_tensor_int8(tensor)
                buf = QuantizedWeightBuffer(data=data.cpu(), scale=scale.cpu(), precision="int8", orig_shape=tuple(tensor.shape), orig_dtype=tensor.dtype)
            elif self.target_precision == "int4":
                data, scale, orig_shape = quantize_tensor_int4(tensor)
                buf = QuantizedWeightBuffer(data=data.cpu(), scale=scale.cpu(), precision="int4", orig_shape=orig_shape, orig_dtype=tensor.dtype)
            elif self.target_precision == "ternary":
                data, scale, orig_shape = quantize_tensor_ternary(tensor)
                buf = QuantizedWeightBuffer(data=data.cpu(), scale=scale.cpu(), precision="ternary", orig_shape=orig_shape, orig_dtype=tensor.dtype)
            else:
                buf = QuantizedWeightBuffer(data=tensor.cpu(), scale=torch.tensor(1.0), precision="float", orig_shape=tuple(tensor.shape), orig_dtype=tensor.dtype)

            region_map[param_name] = buf
            total_bytes += buf.num_bytes

        self._buffers[region_id] = region_map
        return total_bytes

    def load_region(
        self,
        region_id: str,
        device: Optional[torch.device | str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Dequantize and return all tensors for a region."""
        region_map = self._buffers.get(region_id)
        if region_map is None:
            return None
        return {param_name: buf.dequantize(device=device) for param_name, buf in region_map.items()}

    def contains(self, region_id: str) -> bool:
        return region_id in self._buffers

    def evict(self, region_id: str) -> bool:
        return self._buffers.pop(region_id, None) is not None

    def clear(self) -> None:
        self._buffers.clear()
