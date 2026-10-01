"""Format-neutral quantized tensor descriptors and materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    bits: int
    block_shape: tuple[int, ...] | None = None
    scale_dtype: str = "float32"
    zero_point: bool = False
    metadata_bytes_per_block: int = 0


@dataclass(frozen=True)
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    storage_bytes: int
    spec: QuantizationSpec
    payload: Any
    materialized_bytes: int | None = None

    @property
    def compute_bytes(self) -> int:
        if self.materialized_bytes is not None:
            return self.materialized_bytes
        elements = 1
        for dimension in self.shape:
            elements *= int(dimension)
        return elements * 2


def packed_storage_bytes(
    shape: tuple[int, ...],
    bits: int,
    *,
    metadata_bytes: int = 0,
    block_shape: tuple[int, ...] | None = None,
    metadata_bytes_per_block: int = 0,
) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    elements = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("shape dimensions must be non-negative")
        elements *= int(dimension)
    blocks = 1
    if block_shape:
        if len(block_shape) != len(shape):
            raise ValueError("block_shape must have the same rank as shape")
        for dimension, block in zip(shape, block_shape):
            if block <= 0:
                raise ValueError("block dimensions must be positive")
            blocks *= (dimension + block - 1) // block
    return (elements * bits + 7) // 8 + metadata_bytes + blocks * metadata_bytes_per_block


def quantization_spec_from_config(config: Mapping[str, Any]) -> QuantizationSpec | None:
    """Normalize common HF quantization configurations."""
    q = config.get("quantization_config")
    if not isinstance(q, Mapping):
        q = config.get("quantization")
    if not isinstance(q, Mapping):
        return None
    method = str(q.get("quant_method", q.get("mode", ""))).lower()
    bits = int(q.get("bits", 8 if method == "fp8" else 0))
    if bits <= 0:
        return None
    block = q.get("weight_block_size")
    if block is None and q.get("group_size") is not None:
        block = (int(q["group_size"]),)
    block_shape = tuple(int(x) for x in block) if block else None
    return QuantizationSpec(
        scheme=method or "packed",
        bits=bits,
        block_shape=block_shape,
        zero_point=bool(q.get("zero_point", q.get("sym", False)) is False),
    )


class QuantizedMaterializer:
    """Delegate dequantization to a backend while retaining common accounting."""

    def __init__(self, handlers: dict[str, Callable[[QuantizedTensor, Any], Any]]):
        self._handlers = handlers

    def materialize(self, tensor: QuantizedTensor, device: Any = None) -> Any:
        try:
            handler = self._handlers[tensor.spec.scheme]
        except KeyError as exc:
            raise ValueError(
                f"no materializer registered for quantization scheme "
                f"{tensor.spec.scheme!r}"
            ) from exc
        return handler(tensor, device)


class QuantizationRegistry:
    """Map checkpoint tensors to storage and compute representations."""

    def __init__(self) -> None:
        self._schemes: dict[str, QuantizationSpec] = {}

    def register(self, name: str, spec: QuantizationSpec) -> None:
        self._schemes[name] = spec

    def spec_for(self, name: str) -> QuantizationSpec | None:
        return self._schemes.get(name)

    def describe(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        storage_bytes: int | None = None,
        materialized_bytes: int | None = None,
        payload: Any = None,
    ) -> QuantizedTensor | None:
        spec = self.spec_for(name)
        if spec is None:
            return None
        if storage_bytes is None:
            storage_bytes = packed_storage_bytes(
                shape, spec.bits,
                block_shape=spec.block_shape,
                metadata_bytes_per_block=spec.metadata_bytes_per_block,
            )
        return QuantizedTensor(
            name, shape, storage_bytes, spec, payload, materialized_bytes
        )
