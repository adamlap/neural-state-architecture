"""Hardware-neutral residency policy primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ResidentEntry:
    region: str
    size_bytes: int
    next_use: int | None
    priority: float = 0.0


class NextUseEvictionPolicy:
    """Choose victims using future use distance plus explicit priority.

    A region with no known future use is evictable before one that will be used
    soon. Higher priority protects state such as recurrent attention state.
    """

    def choose_victim(
        self,
        resident: Mapping[str, ResidentEntry],
        required_bytes: int,
        capacity_bytes: int,
    ) -> str | None:
        used = sum(entry.size_bytes for entry in resident.values())
        if used + required_bytes <= capacity_bytes:
            return None
        candidates = list(resident.values())
        if not candidates:
            return None
        candidates.sort(
            key=lambda entry: (
                entry.priority,
                float("inf") if entry.next_use is None else entry.next_use,
            )
        )
        return candidates[0].region
