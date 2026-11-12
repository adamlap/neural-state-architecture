"""Organic components: a fixed-size familiarity memory used to gate memory writes and model calls.

``NoveltyGate`` is a linear autoassociative memory (D x D floats, constant) trained online with the delta rule on hashed
character-trigram cues. An event's novelty is the part of its cue the memory cannot reconstruct; the memory then learns the
event, so repeats and near-duplicates (typos, paraphrases that share most trigrams) score low and new content scores high.

Requires NumPy (optional dependency of NSA); importing this module without it raises a clear error on first use.
Research context and measurements: docs/ORGANIC_RESEARCH.md. A familiarity score is lexical: it cannot tell a genuine update whose
wording repeats earlier text from a reminder, so using it to *drop* writes trades recall of such updates for fewer writes/calls.
"""
from __future__ import annotations

import hashlib
from typing import Any

_np: Any = None


def _numpy():
    global _np
    if _np is None:
        try:
            import numpy
        except ImportError as exc:                                    # pragma: no cover - exercised only without numpy
            raise ImportError("nsa.organic needs numpy: pip install numpy") from exc
        _np = numpy
    return _np


def cue(text: str, dim: int = 256):
    """Signed hashed character-trigram vector (similar strings give similar cues), unit norm."""
    np = _numpy()
    v = np.zeros(dim, dtype=np.float32)
    padded = f"  {text.lower()}  "
    for i in range(len(padded) - 2):
        h = int.from_bytes(hashlib.blake2b(padded[i:i + 3].encode(), digest_size=8).digest(), "little")
        v[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class NoveltyGate:
    """Fixed-size familiarity memory. ``observe`` returns the novelty and then learns; ``peek`` does not learn."""

    def __init__(self, dim: int = 256, eta: float = 1.0) -> None:
        if dim <= 0 or not 0.0 < eta <= 1.0:
            raise ValueError("dim must be positive and eta in (0, 1]")
        np = _numpy()
        self.dim, self.eta = int(dim), float(eta)
        self.W = np.zeros((self.dim, self.dim), dtype=np.float32)
        self.floats = self.dim * self.dim
        self.seen = 0

    def peek(self, text: str) -> float:
        x = cue(text, self.dim)
        return float(_numpy().linalg.norm(x - self.W @ x))

    def observe(self, text: str) -> float:
        np = _numpy()
        x = cue(text, self.dim)
        residual = x - self.W @ x
        self.W += self.eta * np.outer(residual, x)
        self.seen += 1
        return float(np.linalg.norm(residual))


__all__ = ["NoveltyGate", "cue"]
