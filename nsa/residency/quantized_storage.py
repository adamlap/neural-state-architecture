"""Adaptive precision and quantized storage for neural regions.

Supports storing cold/warm weights in INT8 or INT4 representations on disk
or host RAM, and dynamically dequantizing them into compute-ready floating point
tensors (FP16/BF16/FP32) upon materialization.

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
    # Ensure even length for 2-element packing
    pad_len = (2 - (flat.numel() % 2)) % 2
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    max_val = torch.max(torch.abs(flat)).clamp(min=1e-8)
    scale = max_val / 7.0
    # Values clamped to [-7, 7] and shifted to unsigned [1, 15], 0 is reserved
    int4_vals = torch.clamp(torch.round(flat / scale), -7, 7).to(torch.int8)
    unsigned_int4 = (int4_vals + 7).to(torch.uint8)  # range 0..14

    # Pack two 4-bit nibbles into one uint8 byte
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

    # Convert back from unsigned [0..14] to signed [-7..7]
    signed = (trimmed.to(dtype=dtype) - 7.0) * scale.to(target_device, dtype=dtype)
    return signed.view(orig_shape).to(dtype)


def quantize_tensor_ternary(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """1.58-bit ternary quantization ({-1, 0, +1}) packed 4 trits per uint8 byte (BitNet b1.58 style)."""
    orig_shape = tuple(tensor.shape)
    flat = tensor.contiguous().view(-1).to(torch.float32)
    # Scale gamma: mean absolute value of weight tensor
    scale = torch.mean(torch.abs(flat)).clamp(min=1e-8)
    # Round and clip to {-1, 0, 1}
    ternary = torch.clamp(torch.round(flat / scale), -1, 1).to(torch.int8)

    # Encode: 0 -> 0, +1 -> 1, -1 -> 2 (2 bits per trit)
    encoded = torch.where(ternary == 1, torch.tensor(1, dtype=torch.uint8, device=tensor.device),
              torch.where(ternary == -1, torch.tensor(2, dtype=torch.uint8, device=tensor.device),
                          torch.tensor(0, dtype=torch.uint8, device=tensor.device)))

    # Pad to multiple of 4
    pad_len = (4 - (encoded.numel() % 4)) % 4
    if pad_len > 0:
        encoded = torch.nn.functional.pad(encoded, (0, pad_len))

    # Pack 4 trits into one uint8 byte
    t0 = encoded[0::4]
    t1 = encoded[1::4]
    t2 = encoded[2::4]
    t3 = encoded[3::4]
    packed = (t0 << 6) | (t1 << 4) | (t2 << 2) | t3

    return packed, scale.to(torch.float32), orig_shape


def dequantize_tensor_ternary(
    packed: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Unpack 2-bit packed ternary bytes back to float tensor scaled by gamma."""
    target_device = device or packed.device
    packed_dev = packed.to(target_device)

    t0 = (packed_dev >> 6) & 0x03
    t1 = (packed_dev >> 4) & 0x03
    t2 = (packed_dev >> 2) & 0x03
    t3 = packed_dev & 0x03

    interleaved = torch.empty(packed.numel() * 4, dtype=torch.uint8, device=target_device)
    interleaved[0::4] = t0
    interleaved[1::4] = t1
    interleaved[2::4] = t2
    interleaved[3::4] = t3

    num_orig = 1
    for s in orig_shape:
        num_orig *= s
    trimmed = interleaved[:num_orig]

    # Map back: 1 -> +1.0, 2 -> -1.0, else 0.0
    ternary_f = torch.where(trimmed == 1, torch.tensor(1.0, dtype=dtype, device=target_device),
                torch.where(trimmed == 2, torch.tensor(-1.0, dtype=dtype, device=target_device),
                            torch.tensor(0.0, dtype=dtype, device=target_device)))

    result = (ternary_f * scale.to(target_device, dtype=dtype)).view(orig_shape).to(dtype)
    return result


def ternary_linear_add(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    scale: torch.Tensor,
    w_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Multiplication-free linear forward pass using pure additions and subtractions."""
    target_device = x.device
    packed_dev = packed_w.to(target_device)
    t0 = (packed_dev >> 6) & 0x03
    t1 = (packed_dev >> 4) & 0x03
    t2 = (packed_dev >> 2) & 0x03
    t3 = packed_dev & 0x03
    interleaved = torch.empty(packed_w.numel() * 4, dtype=torch.uint8, device=target_device)
    interleaved[0::4] = t0
    interleaved[1::4] = t1
    interleaved[2::4] = t2
    interleaved[3::4] = t3

    num_w = w_shape[0] * w_shape[1]
    trimmed = interleaved[:num_w].view(w_shape)

    # Masks for addition vs subtraction
    pos_mask = (trimmed == 1).to(x.dtype)
    neg_mask = (trimmed == 2).to(x.dtype)

    # Pure addition/subtraction: sum(x[pos]) - sum(x[neg])
    pos_contrib = torch.matmul(x, pos_mask.t())
    neg_contrib = torch.matmul(x, neg_mask.t())
    out = (pos_contrib - neg_contrib) * scale.to(target_device, dtype=x.dtype)
    return out


@dataclass
class QuantizedWeightBuffer:
    """Holds a quantized tensor buffer with its decompression metadata."""
    data: torch.Tensor
    scale: torch.Tensor
    precision: str  # "int8" or "int4"
    orig_shape: Tuple[int, ...]
    orig_dtype: torch.dtype

    @property
    def num_bytes(self) -> int:
        return self.data.element_size() * self.data.numel() + self.scale.element_size() * self.scale.numel()

    def dequantize(self, device: Optional[torch.device | str] = None) -> torch.Tensor:
        if self.precision == "int8":
            return dequantize_tensor_int8(self.data, self.scale, dtype=self.orig_dtype, device=device)
        elif self.precision == "int4":
            return dequantize_tensor_int4(self.data, self.scale, self.orig_shape, dtype=self.orig_dtype, device=device)
        elif self.precision == "ternary":
            return dequantize_tensor_ternary(self.data, self.scale, self.orig_shape, dtype=self.orig_dtype, device=device)
        else:
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
                buf = QuantizedWeightBuffer(
                    data=data.cpu(),
                    scale=scale.cpu(),
                    precision="int8",
                    orig_shape=tuple(tensor.shape),
                    orig_dtype=tensor.dtype,
                )
            elif self.target_precision == "int4":
                packed, scale, orig_shape = quantize_tensor_int4(tensor)
                buf = QuantizedWeightBuffer(
                    data=packed.cpu(),
                    scale=scale.cpu(),
                    precision="int4",
                    orig_shape=orig_shape,
                    orig_dtype=tensor.dtype,
                )
            elif self.target_precision == "ternary":
                packed, scale, orig_shape = quantize_tensor_ternary(tensor)
                buf = QuantizedWeightBuffer(
                    data=packed.cpu(),
                    scale=scale.cpu(),
                    precision="ternary",
                    orig_shape=orig_shape,
                    orig_dtype=tensor.dtype,
                )
            else:
                buf = QuantizedWeightBuffer(
                    data=tensor.cpu(),
                    scale=torch.tensor(1.0),
                    precision="float",
                    orig_shape=tuple(tensor.shape),
                    orig_dtype=tensor.dtype,
                )

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

        return {
            param_name: buf.dequantize(device=device)
            for param_name, buf in region_map.items()
        }

    def contains(self, region_id: str) -> bool:
        return region_id in self._buffers

    def evict(self, region_id: str) -> bool:
        return self._buffers.pop(region_id, None) is not None

    def clear(self) -> None:
        self._buffers.clear()