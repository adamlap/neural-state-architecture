"""Wall-clock continuous cognitive-state dynamics.

This module is deliberately separate from ``ContinuousCognitiveEngine``.
CCE schedules authoritative transitions; this field evolves a *soft cognitive
state* between those transitions using real elapsed time. The dynamics are
injected by the caller, so the runtime contains no fabricated cognitive rule.

For a state x and external input u the supplied field computes

    dx/dt = F(x, u)

and the runtime performs explicit numerical integration using the measured
wall-clock delta. Inputs are asynchronous perturbations and may be injected
without starting an LLM inference call.

Hard NSA authority is never part of this mutable state container. A caller
that wants to commit authoritative state must go through the existing NSA
transition/substrate boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Callable, Generic, List, Optional, TypeVar

import torch

InputT = TypeVar("InputT")
Field = Callable[[torch.Tensor, Optional[InputT]], torch.Tensor]
InputReducer = Callable[[List[InputT]], Optional[InputT]]


def _last_input(inputs: List[InputT]) -> Optional[InputT]:
    """Default policy: deliver the newest pending event to the field."""
    return inputs[-1] if inputs else None


@dataclass(frozen=True)
class ContinuousFieldStatus:
    enabled: bool
    running: bool
    integration_count: int
    elapsed_seconds: float
    last_dt: float
    last_error: Optional[str]
    pending_inputs: int


class ContinuousStateField(Generic[InputT]):
    """A real-time numerical integrator for an injected cognitive state field.

    ``step_now()`` advances from the previous monotonic timestamp to the
    current timestamp. ``start()`` repeatedly performs the same operation at
    an implementation cadence, but integration always uses measured elapsed
    time rather than assuming that cadence is the cognitive clock.

    ``input_reducer`` makes asynchronous event handling explicit. The default
    is last-event-wins, while callers that must preserve every event can supply
    a reducer that combines the pending events into one field input.
    """

    def __init__(
        self,
        state: torch.Tensor,
        field: Field[InputT],
        *,
        integration_cadence_seconds: float = 0.01,
        enabled: bool = False,
        fail_closed: bool = True,
        input_reducer: Optional[InputReducer[InputT]] = None,
    ) -> None:
        if state.ndim == 0:
            raise ValueError("state must have at least one dimension")
        if integration_cadence_seconds <= 0:
            raise ValueError("integration_cadence_seconds must be > 0")
        self._state = state.detach().clone()
        self._field = field
        self._cadence = float(integration_cadence_seconds)
        self._enabled = bool(enabled)
        self._fail_closed = bool(fail_closed)
        self._input_reducer = input_reducer or _last_input
        self._running = False
        self._integration_count = 0
        self._elapsed = 0.0
        self._last_dt = 0.0
        self._last_error: Optional[str] = None
        self._last_time: Optional[float] = None
        self._pending_inputs: List[InputT] = []
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None

    @property
    def state(self) -> torch.Tensor:
        with self._lock:
            return self._state.detach().clone()

    def status(self) -> ContinuousFieldStatus:
        with self._lock:
            return ContinuousFieldStatus(
                enabled=self._enabled,
                running=self._running,
                integration_count=self._integration_count,
                elapsed_seconds=self._elapsed,
                last_dt=self._last_dt,
                last_error=self._last_error,
                pending_inputs=len(self._pending_inputs),
            )

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
        if not enabled:
            self.stop()

    def inject(self, value: InputT) -> None:
        """Queue an asynchronous external perturbation."""
        with self._lock:
            self._pending_inputs.append(value)

    def step_now(self, now: Optional[float] = None) -> bool:
        """Integrate once using actual elapsed wall-clock time."""
        timestamp = monotonic() if now is None else float(now)
        with self._lock:
            if not self._enabled:
                return False
            previous = self._last_time
            if previous is None:
                self._last_time = timestamp
                return False
            dt = max(0.0, timestamp - previous)
            self._last_time = timestamp
            state = self._state
            inputs = self._pending_inputs
            self._pending_inputs = []

        try:
            external = self._input_reducer(inputs)
            derivative = self._field(state, external)
            if derivative.shape != state.shape:
                raise ValueError(
                    f"field returned shape {tuple(derivative.shape)}; expected {tuple(state.shape)}"
                )
            if not torch.isfinite(derivative).all():
                raise ValueError("field returned non-finite derivative")
            next_state = state + derivative * dt
            if not torch.isfinite(next_state).all():
                raise ValueError("integration produced non-finite state")
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                if self._fail_closed:
                    self._enabled = False
                    self._stop.set()
            return False

        with self._lock:
            self._state = next_state.detach()
            self._integration_count += 1
            self._elapsed += dt
            self._last_dt = dt
            self._last_error = None
        return True

    def start(self) -> bool:
        with self._lock:
            if not self._enabled or self._running:
                return False
            self._stop.clear()
            self._last_time = monotonic()
            self._running = True
            self._thread = Thread(target=self._run, name="nsa-continuous-field", daemon=True)
            thread = self._thread
        thread.start()
        return True

    def stop(self, timeout: Optional[float] = None) -> bool:
        with self._lock:
            thread = self._thread
            was_running = self._running
            self._stop.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            self._running = False
            if self._thread is thread:
                self._thread = None
        return was_running

    def _run(self) -> None:
        while not self._stop.wait(self._cadence):
            with self._lock:
                if not self._enabled:
                    break
            self.step_now()
        with self._lock:
            self._running = False
            self._thread = None
