"""Hardware-neutral streaming residency execution primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


@dataclass
class ResidencyTelemetry:
    bytes_loaded: int = 0
    regions_loaded: int = 0
    regions_evicted: int = 0
    load_seconds: float = 0.0
    compute_seconds: float = 0.0
    peak_resident_bytes: int = 0
    events: list[dict[str, float | int | str]] = field(default_factory=list)


class RegionStore(Generic[T]):
    """Minimal storage interface; implementations can use disk, RAM, NVMe, etc."""

    def load(self, region: str) -> T:
        raise NotImplementedError

    def size(self, region: str) -> int:
        raise NotImplementedError


class InMemoryRegionStore(RegionStore[T]):
    def __init__(self, regions: dict[str, T], sizes: dict[str, int] | None = None):
        self._regions = regions
        self._sizes = sizes or {name: 0 for name in regions}

    def load(self, region: str) -> T:
        return self._regions[region]

    def size(self, region: str) -> int:
        return self._sizes.get(region, 0)


class StreamingExecutor(Generic[T]):
    """Execute an ordered region stream under a hard byte budget.

    Backend-specific code is supplied as callbacks.  No device APIs are
    assumed, so the same scheduler can drive CPU, GPU, NPU or custom kernels.
    """

    def __init__(self, store: RegionStore[T], budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.store = store
        self.budget_bytes = budget_bytes
        self.telemetry = ResidencyTelemetry()

    def run(self, regions: list[str], compute: Callable[[str, T], None]) -> ResidencyTelemetry:
        resident: dict[str, T] = {}
        resident_bytes = 0

        for name in regions:
            size = self.store.size(name)
            if size > self.budget_bytes:
                raise MemoryError(
                    f"region {name!r} requires {size} bytes, budget is {self.budget_bytes}"
                )

            started = perf_counter()
            value = self.store.load(name)
            self.telemetry.load_seconds += perf_counter() - started
            self.telemetry.bytes_loaded += size
            self.telemetry.regions_loaded += 1
            resident[name] = value
            resident_bytes += size
            self.telemetry.peak_resident_bytes = max(
                self.telemetry.peak_resident_bytes, resident_bytes
            )

            started = perf_counter()
            compute(name, value)
            self.telemetry.compute_seconds += perf_counter() - started

            # Sequential execution has no need to retain completed regions.
            resident.pop(name, None)
            resident_bytes -= size
            self.telemetry.regions_evicted += 1
            self.telemetry.events.append({
                "region": name,
                "size": size,
                "resident_bytes": resident_bytes,
            })

        return self.telemetry
