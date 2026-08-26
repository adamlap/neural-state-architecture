"""Public continuous cognitive execution primitives.

This module is the canonical wall-clock scheduler for NSA.  It owns scheduling
only; the callable supplied as ``step`` remains the authoritative state
transition function.  Cognition, policy, capability checks, and state commits
therefore remain outside the scheduler and cannot be bypassed by continuous
execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from time import monotonic
from typing import Callable, Generic, Optional, TypeVar

S = TypeVar("S")


@dataclass(frozen=True)
class CCEStatus:
    """Immutable observability snapshot of continuous execution."""

    enabled: bool
    running: bool
    tick_count: int
    last_tick_monotonic: Optional[float]
    last_error: Optional[str]


class ContinuousCognitiveEngine(Generic[S]):
    """Run persistent NSA state transitions on an optional wall-clock loop.

    ``step`` is the sole transition authority.  The scheduler never mutates
    state directly and never grants capabilities.  Execution is opt-in and
    fail-closed by default: a transition exception freezes state and disables
    automatic ticks until the caller explicitly enables execution again.
    """

    def __init__(
        self,
        state: S,
        step: Callable[[S], S],
        *,
        interval_seconds: float = 0.1,
        enabled: bool = False,
        fail_closed: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        self._state = state
        self._step = step
        self._interval = float(interval_seconds)
        self._enabled = bool(enabled)
        self._fail_closed = bool(fail_closed)
        self._running = False
        self._tick_count = 0
        self._last_tick: Optional[float] = None
        self._last_error: Optional[str] = None
        self._lock = Lock()
        self._tick_lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None

    @property
    def state(self) -> S:
        with self._lock:
            return self._state

    def set_state(self, state: S) -> None:
        """Replace scheduler state without executing a transition."""
        with self._lock:
            self._state = state

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
        """Execute exactly one authoritative transition when enabled."""
        if not self._tick_lock.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if not self._enabled:
                    return False
                current = self._state
            try:
                next_state = self._step(current)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    if self._fail_closed:
                        self._enabled = False
                        self._stop.set()
                return False
            with self._lock:
                self._state = next_state
                self._tick_count += 1
                self._last_tick = monotonic()
                self._last_error = None
            return True
        finally:
            self._tick_lock.release()

    def start(self) -> bool:
        """Start wall-clock execution when explicitly enabled."""
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
        """Stop wall-clock execution without changing committed state."""
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


__all__ = ["CCEStatus", "ContinuousCognitiveEngine"]
