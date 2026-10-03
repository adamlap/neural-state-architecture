from __future__ import annotations

from dataclasses import dataclass
from nsa.core.state import CanonicalState


@dataclass(frozen=True)
class StrategicUpdatePolicy:
    """Decide when a strategic field should be recomputed.

    The policy observes soft-state changes and evaluation cadence. It does not
    mutate canonical state or inspect/alter capability authority.
    """
    risk_threshold: float = 0.10
    uncertainty_threshold: float = 0.10
    step_interval: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("risk_threshold", self.risk_threshold),
            ("uncertainty_threshold", self.uncertainty_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.step_interval < 1:
            raise ValueError("step_interval must be >= 1")

    def should_update(
        self,
        previous: CanonicalState | None,
        current: CanonicalState,
    ) -> bool:
        if previous is None:
            return True
        if current.step - previous.step >= self.step_interval:
            return True
        if abs(current.soft.risk - previous.soft.risk) >= self.risk_threshold:
            return True
        if abs(current.soft.uncertainty - previous.soft.uncertainty) >= self.uncertainty_threshold:
            return True
        return False
