"""Sensory Ingress and Continuous Perturbation Adapter for CCE.

Implements the text input adapter from CCE Phase 6 (Sensory Interfaces).
Maps unstructured text and asynchronous events to continuous perturbation
vectors u(t) in R^d that drive state updates inside PersistentCognitiveState.

No semantic keyword extraction is performed here.  The raw state tensor X(t)
carries the integrated effect of all past inputs; the LLM receives the numeric
state directly and is responsible for authentic interpretation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class SensoryEvent:
    """A timestamped sensory event logged in the ingress ring-buffer."""

    source: str
    content: str
    timestamp_utc: float
    importance: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CCESensoryIngress:
    """Maps external text/sensor streams to bounded perturbation vectors u(t) in R^d.

    Design notes
    ------------
    * Deterministic SHA-256 projection: identical text + source produces
      identical vector, enabling reproducible ablation studies.
    * No keyword or topic extraction - the plan specifies that the continuous
      state tensor itself carries meaning; keyword extraction would shortcut
      the architectural hypothesis being validated.
    * Importance [0, 1] scales the perturbation magnitude linearly.
    * Ring-buffer of the last 100 events kept for provenance inspection.
    """

    def __init__(self, dimension: int = 4, scale: float = 0.5) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        self.dimension = int(dimension)
        self.scale = float(scale)
        self._history: List[SensoryEvent] = []

    def encode_text_to_perturbation(
        self,
        text: str,
        *,
        source: str = "external",
        importance: float = 0.5,
    ) -> Tuple[torch.Tensor, SensoryEvent]:
        """Project raw input text to a continuous perturbation vector.

        Args:
            text:       Raw input string (chat turn, sensor reading, etc.)
            source:     Named provenance tag (e.g. "openwebui_chat").
            importance: Scalar in [0, 1] that scales perturbation magnitude.

        Returns:
            (perturbation_tensor, sensory_event)
        """
        importance = max(0.0, min(1.0, float(importance)))
        event = SensoryEvent(
            source=source,
            content=text[:500],
            timestamp_utc=time.time(),
            importance=importance,
        )
        self._history.append(event)
        if len(self._history) > 100:
            self._history.pop(0)

        clean = text.strip()
        coords: List[float] = []
        for i in range(self.dimension):
            seed = f"{source}:{i}:{clean[:256]}"
            h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
            val = ((h / 0xFFFFFFFF) * 2.0 - 1.0) * self.scale * importance
            coords.append(val)

        perturbation = torch.tensor(coords, dtype=torch.float32)
        return perturbation, event

    @property
    def recent_events(self) -> List[SensoryEvent]:
        return list(self._history)


__all__ = ["SensoryEvent", "CCESensoryIngress"]
