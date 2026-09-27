"""Format-neutral quantized tensor descriptors and materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    bits: int
    block_shape: tuple[int, ...] | None = None
    scale_dtype: str = "float32"
    zero_point: bool = False


@dataclass(frozen=True)
class QuantizedTensor:
    name: str
    shape: tuple[int, ...]
    storage_bytes: int
    spec: QuantizationSpec
    payload: Any


def packed_storage_bytes(
    shape: tuple[int, ...],
    bits: int,
    *,
    metadata_bytes: int = 0,
) -> int:
    if bits <= 0:
        raise ValueError("bits must be positive")
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    return (elements * bits + 7) // 8 + metadata_bytes


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
