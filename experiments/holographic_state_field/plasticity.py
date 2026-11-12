"""Small online competitive-Hebbian map for self-organisation experiments.

This is an experimental representation learner, not part of the NSA runtime.
It receives only vectors: family labels are deliberately absent from the
learning API. Winner/neighbour updates are local, followed by row
normalisation to keep prototype magnitudes bounded.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class CompetitiveHebbianMap:
    """A 1-D self-organising codebook with local competitive plasticity."""

    def __init__(self, *, units: int, dimension: int, seed: int = 7,
                 learning_rate: float = 0.2, neighbourhood: float = 1.0) -> None:
        if units < 2 or dimension < 2:
            raise ValueError("units and dimension must both be >= 2")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if neighbourhood < 0:
            raise ValueError("neighbourhood must be non-negative")
        self.units = int(units)
        self.dimension = int(dimension)
        self.learning_rate = float(learning_rate)
        self.neighbourhood = float(neighbourhood)
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=(units, dimension)).astype(np.float32)
        self._weights = self._normalise_rows(weights)
        self.updates = 0

    @staticmethod
    def _normalise_rows(values: FloatArray) -> FloatArray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, np.float32(1e-12))

    @property
    def weights(self) -> FloatArray:
        """Return a copy so callers cannot mutate the learner's state."""
        return self._weights.copy()

    @property
    def memory_bytes(self) -> int:
        return int(self._weights.nbytes)

    def _cue(self, vector: FloatArray) -> FloatArray:
        cue = np.asarray(vector, dtype=np.float32).reshape(-1)
        if cue.size != self.dimension or not np.all(np.isfinite(cue)):
            raise ValueError("cue must be finite and match the input dimension")
        norm = float(np.linalg.norm(cue))
        if norm <= 0:
            raise ValueError("cue must have non-zero norm")
        return cue / np.float32(norm)

    def winner(self, vector: FloatArray) -> int:
        cue = self._cue(vector)
        return int(np.argmax(self._weights @ cue))

    def learn_one(self, vector: FloatArray) -> int:
        """Apply a local winner/neighbour Hebbian update without labels."""
        cue = self._cue(vector)
        winner = int(np.argmax(self._weights @ cue))
        positions = np.arange(self.units, dtype=np.float32)
        distance = np.abs(positions - np.float32(winner))
        if self.neighbourhood == 0:
            influence = (distance == 0).astype(np.float32)
        else:
            influence = np.exp(
                -(distance ** 2) / (2.0 * self.neighbourhood ** 2)
            ).astype(np.float32)
        rates = (np.float32(self.learning_rate) * influence)[:, None]
        # Competitive Hebbian/SOM update: active units move toward the cue.
        self._weights += rates * (cue[None, :] - self._weights)
        self._weights = self._normalise_rows(self._weights)
        self.updates += 1
        return winner

    def fit(self, vectors: NDArray, *, epochs: int = 1, seed: int = 7) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        data = np.asarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (n_samples, dimension)")
        if data.shape[0] == 0 or not np.all(np.isfinite(data)):
            raise ValueError("vectors must be non-empty and finite")
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for index in rng.permutation(data.shape[0]):
                self.learn_one(data[index])

    def predict(self, vectors: NDArray) -> NDArray[np.int64]:
        data = np.asarray(vectors, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (n_samples, dimension)")
        if not np.all(np.isfinite(data)):
            raise ValueError("vectors must be finite")
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("vectors must be non-zero")
        normalised = data / norms
        return np.argmax(normalised @ self._weights.T, axis=1).astype(np.int64)
