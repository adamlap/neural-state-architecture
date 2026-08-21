from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class DynamicalState:
    """Small bounded multi-timescale state for R2 experiments."""
    fast: float = 0.0
    medium: float = 0.0
    slow: float = 0.0

    def tick(self, drive: float = 0.0, dt: float = 0.1) -> "DynamicalState":
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.fast = math.tanh(self.fast + dt * (-1.5 * self.fast + drive))
        self.medium = math.tanh(self.medium + dt * (-0.35 * self.medium + 0.25 * self.fast))
        self.slow = math.tanh(self.slow + dt * (-0.08 * self.slow + 0.08 * self.medium))
        return self

    def vector(self) -> tuple[float, float, float]:
        return self.fast, self.medium, self.slow
