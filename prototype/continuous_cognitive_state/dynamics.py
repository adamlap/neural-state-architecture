from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DynamicsParameters:
    """Runtime/configurable continuous dynamics; coefficients are not action policy."""
    fast_decay: float = 1.5
    medium_decay: float = 0.35
    medium_coupling: float = 0.25
    slow_decay: float = 0.08
    slow_coupling: float = 0.08


@dataclass
class DynamicalState:
    """Bounded multi-timescale state evolved at every CCE tick."""
    fast: float = 0.0
    medium: float = 0.0
    slow: float = 0.0
    parameters: DynamicsParameters = DynamicsParameters()

    def tick(self, drive: float = 0.0, dt: float = 0.1) -> "DynamicalState":
        if dt <= 0:
            raise ValueError("dt must be positive")
        p = self.parameters
        self.fast = math.tanh(self.fast + dt * (-p.fast_decay * self.fast + drive))
        self.medium = math.tanh(self.medium + dt * (-p.medium_decay * self.medium + p.medium_coupling * self.fast))
        self.slow = math.tanh(self.slow + dt * (-p.slow_decay * self.slow + p.slow_coupling * self.medium))
        return self

    def vector(self) -> tuple[float, float, float]:
        return self.fast, self.medium, self.slow
