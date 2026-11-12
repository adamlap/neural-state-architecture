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


class TrailingField:
    """Leaky trace of a continuous input stream: ``z <- lam * z + (1 - lam) * x``.

    The trace keeps recent input present for a while, so a *delayed* outcome can bind to the cue that preceded it (trace conditioning). A trace binds
    only while it lasts and only while earlier events have decayed: roughly one to two time constants (docs/ORGANIC_RESEARCH.md, Experiment 2).
    """

    def __init__(self, dim: int, lam: float = 0.95) -> None:
        if dim <= 0 or not 0.0 < lam < 1.0:
            raise ValueError("dim must be positive and lam in (0, 1)")
        self.lam = float(lam)
        self.z = _numpy().zeros(int(dim), dtype=_numpy().float32)

    def step(self, x):
        np = _numpy()
        self.z = self.lam * self.z + (1.0 - self.lam) * np.asarray(x, dtype=np.float32)
        return self.z

    def key(self):
        np = _numpy()
        n = float(np.linalg.norm(self.z))
        return self.z / n if n > 1e-8 else self.z.copy()


class StreamLearner:
    """Learns signal -> outcome associations from a continuous stream with ONE fixed vector.

    Pipeline: trailing field -> fixed random expansion with k-winners-take-all (a sparse code) -> a single memory vector ``M`` updated by the delta rule
    only when the prediction error exceeds ``theta`` (surprise-gated writes). Nothing is stored per event; memory is ``sparse_dim`` floats.
    ``predict()`` uses the current field state; ``learn(outcome)`` binds the outcome to the *live* field state, i.e. to whatever the trace still holds.
    ``uncertainty`` is an exponential moving average of the absolute prediction error on outcomes (a calibrated-by-construction surprise signal).
    """

    def __init__(self, dim: int, *, lam: float = 0.95, sparse_dim: int = 1024, sparsity: float = 0.05, eta: float = 0.5, theta: float = 0.15,
                 seed: int = 0) -> None:
        np = _numpy()
        self.field = TrailingField(dim, lam)
        self.W = np.random.default_rng(seed + 9001).standard_normal((sparse_dim, dim)).astype(np.float32)
        self.k = max(1, int(round(sparse_dim * sparsity)))
        self.M = np.zeros(sparse_dim, dtype=np.float32)
        self.eta, self.theta = float(eta), float(theta)
        self.floats = sparse_dim
        self.uncertainty = 1.0
        self.updates = self.outcomes = 0
        self.seen = np.zeros(sparse_dim, dtype=np.float32)     # how often each code unit has taken part in a learning step (written or not)

    def _code(self):
        np = _numpy()
        drive = self.W @ self.field.key()
        code = np.zeros_like(drive)
        code[np.argpartition(drive, -self.k)[-self.k:]] = 1.0
        return code / np.sqrt(self.k)

    def observe(self, x) -> None:
        self.field.step(x)

    def predict(self) -> float:
        return float(self.M @ self._code())

    def familiarity(self) -> float:
        """0..1: how well the current input's active units have been seen in learning steps (0 = never seen; ~1 = seen many times).

        Distinct from the prediction: an input learned to mean "nothing happens" has a prediction near 0 but high familiarity; a never-seen input has both near 0.
        """
        active = self._code() > 0
        n = float(self.seen[active].mean()) if active.any() else 0.0
        return float(1.0 - _numpy().exp(-n / 3.0))

    def learn(self, outcome: float, *, enabled: bool = True) -> float:
        """Bind ``outcome`` to the current field state; returns the prediction error (before the update)."""
        code = self._code()
        error = float(outcome) - float(self.M @ code)
        self.outcomes += 1
        self.seen += (code > 0)
        self.uncertainty += 0.1 * (min(1.0, abs(error)) - self.uncertainty)
        if enabled and abs(error) > self.theta:
            self.M += self.eta * error * code / float(code @ code)
            self.updates += 1
        return error


__all__ = ["NoveltyGate", "TrailingField", "StreamLearner", "cue"]
