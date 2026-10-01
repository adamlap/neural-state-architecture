"""Portable execution contract between NSA and compute hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    kind: str
    memory_bytes: int
    compute_units: int = 0
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DeviceBuffer:
    """Backend-owned value together with residency accounting metadata."""
    value: Any
    device: DeviceInfo
    size_bytes: int


@dataclass
class TransferMetrics:
    moves: int = 0
    bytes_moved: int = 0
    releases: int = 0


def value_nbytes(value: Any) -> int:
    """Best-effort byte size without depending on a tensor library."""
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 0


class ExecutionBackend(Protocol):
    """Contract implemented by CPU/GPU/NPU/custom backends."""

    def device_info(self) -> DeviceInfo:
        ...

    def move(self, value: Any, device: DeviceInfo) -> Any:
        ...

    def execute(self, operation: str, inputs: tuple[Any, ...], **kwargs: Any) -> Any:
        ...

    def release(self, value: Any) -> None:
        ...


class CpuBackend:
    """Reference backend used for correctness tests and portable fallback."""

    def __init__(self, memory_bytes: int = 0):
        self._info = DeviceInfo("cpu", "cpu", memory_bytes)
        self.metrics = TransferMetrics()

    def device_info(self) -> DeviceInfo:
        return self._info

    def move(self, value: Any, device: DeviceInfo) -> Any:
        self.metrics.moves += 1
        if device != self._info:
            self.metrics.bytes_moved += value_nbytes(value)
        return value

    def execute(self, operation: str, inputs: tuple[Any, ...], **kwargs: Any) -> Any:
        if operation == "identity":
            return inputs[0]
        if operation == "add":
            return inputs[0] + inputs[1]
        if operation == "matmul":
            return inputs[0] @ inputs[1]
        raise ValueError(f"unsupported CPU operation {operation!r}")

    def release(self, value: Any) -> None:
        self.metrics.releases += 1
