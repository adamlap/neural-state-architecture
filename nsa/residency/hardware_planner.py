"""Select a residency strategy from hardware capabilities, not model identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .model_plan import ModelResidencyPlan
from .model_analysis import ResidencyAnalysis, analyze_plan


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    kind: str
    memory_bytes: int
    capabilities: FrozenSet[str] = frozenset()
    preferred_quantization: tuple[str, ...] = ()


@dataclass(frozen=True)
class HardwarePlan:
    hardware: HardwareProfile
    analysis: ResidencyAnalysis
    requires_streaming: bool
    supported_quantization: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "hardware": self.hardware.name,
            "kind": self.hardware.kind,
            "memory_bytes": self.hardware.memory_bytes,
            "requires_streaming": self.requires_streaming,
            "supported_quantization": self.supported_quantization,
            "analysis": self.analysis.to_dict(),
        }


def plan_for_hardware(
    plan: ModelResidencyPlan,
    hardware: HardwareProfile,
) -> HardwarePlan:
    analysis = analyze_plan(plan, residency_budget_bytes=hardware.memory_bytes)
    preferred = next(
        (scheme for scheme in hardware.preferred_quantization if scheme),
        None,
    )
    return HardwarePlan(
        hardware=hardware,
        analysis=analysis,
        requires_streaming=plan.storage_bytes > hardware.memory_bytes,
        supported_quantization=preferred,
    )
