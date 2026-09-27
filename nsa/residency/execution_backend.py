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


class ExecutionBackend(Protocol):
    """Minimal contract implemented by CPU/GPU/NPU/custom backends."""

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

    def device_info(self) -> DeviceInfo:
        return self._info

    def move(self, value: Any, device: DeviceInfo) -> Any:
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
        del value
