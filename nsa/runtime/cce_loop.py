"""Closed-loop CCE invocation orchestration.

This module connects continuous salience to an inference backend without giving
that backend authority over NSA state. It is deliberately orchestration-only:
a salient event may request cognition, but the resulting text is an observation
/proposal that must be handled by the caller's existing governed transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nsa.runtime.cce_salience import AdaptiveSalienceGate, SalienceObservation
from nsa.runtime.inference.base import InferenceBackend


@dataclass(frozen=True)
class CognitiveInvocation:
    """Immutable record of one salience decision and optional inference."""

    triggered: bool
    score: float
    threshold: float
    baseline: float
    response: Optional[str]


class ClosedLoopCognitiveInvoker:
    """Invoke an inference backend only when adaptive salience fires.

    The backend is never passed a mutable NSA state object and its response is
    never applied automatically. ``on_response`` is an explicit observation
    hook owned by the caller, preserving the governance boundary.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        gate: Optional[AdaptiveSalienceGate] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.backend = backend
        self.gate = gate or AdaptiveSalienceGate()
        self.on_response = on_response
        self.invocation_count = 0

    def observe(
        self,
        observation: SalienceObservation,
        prompt: str,
        *,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> CognitiveInvocation:
        decision = self.gate.observe(observation)
        if not decision.triggered:
            return CognitiveInvocation(
                triggered=False,
                score=decision.score,
                threshold=decision.threshold,
                baseline=decision.baseline,
                response=None,
            )

        result = self.backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.invocation_count += 1
        if self.on_response is not None:
            self.on_response(result.text)
        return CognitiveInvocation(
            triggered=True,
            score=decision.score,
            threshold=decision.threshold,
            baseline=decision.baseline,
            response=result.text,
        )


__all__ = ["ClosedLoopCognitiveInvoker", "CognitiveInvocation"]
