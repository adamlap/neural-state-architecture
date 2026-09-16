"""Tool/capability descriptors used at the CCE boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nsa.core.capabilities import TrustTier


@dataclass(frozen=True)
class ToolSpec:
    """Describes an executable capability without providing execution authority."""

    name: str
    capability: str
    risk: float = 0.0
    reversible: bool = True
    side_effect: bool = False
    required_trust: TrustTier = TrustTier.T2_REVERSIBLE
    target_type: str | None = None
    provenance_required: bool = False
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.capability.strip():
            raise ValueError("capability must be non-empty")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if self.required_trust == TrustTier.T2_REVERSIBLE and not self.reversible:
            object.__setattr__(self, "required_trust", TrustTier.T3_SIDE_EFFECTS)

    def to_action_defaults(self) -> dict[str, Any]:
        """Return safe defaults for converting a model proposal to an action."""
        return {
            "action_id": self.name,
            "risk": self.risk,
            "reversible": self.reversible,
            "required_capabilities": (self.capability,),
        }


class ToolRegistry:
    """Immutable-ish descriptor registry; it does not execute tools."""

    def __init__(self, tools: tuple[ToolSpec, ...] = ()) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def require(self, name: str) -> ToolSpec:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def all(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())


__all__ = ["ToolRegistry", "ToolSpec"]
