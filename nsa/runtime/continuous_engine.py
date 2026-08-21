"""Opt-in continuous execution for the NSA cognitive substrate.

The engine deliberately separates *time* from *cognition*: the existing
``CognitiveDynamicsSubstrate.step`` remains the authority for state
transitions and safety decisions. CCE only schedules those transitions.

This makes continuous execution testable and reversible without changing the
behaviour of existing tick-driven callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Generic, Optional, TypeVar

S = TypeVar("S")


@dataclass(frozen=True)
class CCEStatus:
    """Immutable observability snapshot of the continuous engine."""

    enabled: bool
    running: bool
    tick_count: int
    last_tick_monotonic: Optional[float]
    last_error: Optional[str]


class ContinuousCognitiveEngine(Generic[S]):
    """Schedule persistent NSA state transitions on a wall-clock loop.

    ``step`` is supplied by the caller and should perform exactly one
    authoritative NSA cognitive transition. CCE never mutates state itself.

    Continuous execution is opt-in. ``enabled=False`` means ``start()`` is a
    no-op and ``tick()`` does not invoke the transition callback.
    """

    def __init__(
        self,
        state: S,
        step: Callable[[S], S],
        *,
        interval_seconds: float = 0.1,
        enabled: bool = False,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._state = state
        self._step = step
        self._interval = float(interval_seconds)
        self._enabled = bool(enabled)
        self._running = False
        self._tick_count = 0
        self._last_tick: Optional[float] = None
        self._last_error: Optional[str] = None
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None

    @property
    def state(self) -> S:
        with self._lock:
            return self._state

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable future ticks; disabling also stops a running loop."""
        with self._lock:
            self._enabled = bool(enabled)
        if not enabled:
            self.stop()

    def status(self) -> CCEStatus:
        with self._lock:
            return CCEStatus(
                enabled=self._enabled,
                running=self._running,
                tick_count=self._tick_count,
                last_tick_monotonic=self._last_tick,
                last_error=self._last_error,
            )

    def tick(self) -> bool:
        """Execute one transition if enabled; return whether a tick occurred."""
        with self._lock:
            if not self._enabled:
                return False
            current = self._state

        try:
            next_state = self._step(current)
        except Exception as exc:  # keep the scheduler alive; expose the error
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            return False

        with self._lock:
            self._state = next_state
            self._tick_count += 1
            self._last_tick = monotonic()
            self._last_error = None
        return True

    def start(self) -> bool:
        """Start the background wall-clock loop when enabled."""
        with self._lock:
            if not self._enabled or self._running:
                return False
            self._stop.clear()
            self._running = True
            self._thread = Thread(target=self._run, name="nsa-cce", daemon=True)
            thread = self._thread
        thread.start()
        return True

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Stop the background loop without changing the current state."""
        with self._lock:
            thread = self._thread
            was_running = self._running
            self._stop.set()
        if thread is not None and thread is not Thread.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            self._running = False
            if self._thread is thread:
                self._thread = None
        return was_running

    def _run(self) -> None:
        deadline = monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    enabled = self._enabled
                if not enabled:
                    break

                now = monotonic()
                if now >= deadline:
                    self.tick()
                    deadline = now + self._interval
                else:
                    self._stop.wait(deadline - now)
        finally:
            with self._lock:
                self._running = False
                self._thread = None
