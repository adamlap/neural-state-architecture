"""Two-rate continuous runtime supervision.

The supervisor separates two explicitly testable processes:

* high-rate soft-state maintenance (the proposed "phantom processing" primitive);
* lower-rate live model heartbeats that keep the inference model active.

This is an engineering hypothesis about persistent computation, not a claim
that the mechanism creates consciousness. All state remains explicit and the
hard authority field is preserved by the trusted runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock, Thread
from time import monotonic, sleep
from typing import Optional

from nsa.runtime.continuous_engine import ContinuousCognitiveEngine
from nsa.runtime.phantom_maintenance import maintain
from nsa.runtime.typed_runtime import NSATypedRuntime


@dataclass(frozen=True)
class ContinuousRuntimeStatus:
    maintenance_ticks: int
    model_ticks: int
    state_step: int
    running: bool
    hard_authority_unchanged: bool
    last_error: Optional[str]


class ContinuousRuntimeSupervisor:
    """Run persistent soft-state maintenance and live model heartbeats."""

    def __init__(
        self,
        runtime: NSATypedRuntime,
        *,
        maintenance_interval: float = 0.1,
        model_interval: float = 1.0,
        model_prompt: str = "Provide one short sentence describing your current task context.",
    ) -> None:
        if maintenance_interval <= 0 or model_interval <= 0:
            raise ValueError("intervals must be > 0")
        self.runtime = runtime
        self.maintenance_interval = maintenance_interval
        self.model_interval = model_interval
        self.model_prompt = model_prompt
        self._lock = RLock()
        self._stop = Event()
        self._maintenance_thread: Optional[Thread] = None
        self._maintenance_ticks = 0
        self._model_ticks = 0
        self._last_error: Optional[str] = None
        self._authority = runtime.activation.state.authority_state.detach().clone()
        self._model_engine = ContinuousCognitiveEngine(
            runtime,
            self._model_step,
            interval_seconds=model_interval,
            enabled=True,
            fail_closed=True,
        )

    def _model_step(self, runtime: NSATypedRuntime) -> NSATypedRuntime:
        with self._lock:
            runtime.generate(self.model_prompt, max_tokens=32, temperature=0.0)
            self._model_ticks += 1
            return runtime

    def _maintenance_loop(self) -> None:
        last = monotonic()
        try:
            while not self._stop.wait(self.maintenance_interval):
                now = monotonic()
                elapsed = max(0.0, now - last)
                last = now
                with self._lock:
                    result = maintain(self.runtime, elapsed_seconds=elapsed)
                    self._maintenance_ticks += 1
                    if not result.hard_authority_unchanged:
                        raise RuntimeError("continuous maintenance changed hard authority")
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._stop.set()

    def start(self) -> None:
        self._stop.clear()
        self._maintenance_thread = Thread(target=self._maintenance_loop, name="nsa-maintenance", daemon=True)
        self._maintenance_thread.start()
        if not self._model_engine.start():
            self._stop.set()
            raise RuntimeError("model heartbeat engine failed to start")

    def stop(self, timeout: float = 30.0) -> ContinuousRuntimeStatus:
        self._stop.set()
        self._model_engine.stop(timeout=timeout)
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=timeout)
        authority_unchanged = bool((self.runtime.activation.state.authority_state == self._authority).all().item())
        return ContinuousRuntimeStatus(
            maintenance_ticks=self._maintenance_ticks,
            model_ticks=self._model_ticks,
            state_step=self.runtime.activation.state.temporal_state.step_index,
            running=False,
            hard_authority_unchanged=authority_unchanged,
            last_error=self._last_error or self._model_engine.status().last_error,
        )
