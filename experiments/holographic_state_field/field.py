"""Fixed-size, image-like associative memory using distributed superposition.

A key deterministically generates a 2-D bipolar pattern. Its value generates
a channel code. Writing adds their outer product to one persistent H x W x C
field. Retrieval correlates the query pattern with the field and decodes the
reconstructed channel vector against a fixed vocabulary. No per-record table
is retained by the field.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


def _seed_for(seed: int, namespace: str, value: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _bipolar_pattern(seed: int, namespace: str, value: str, dimension: int) -> FloatArray:
    rng = np.random.default_rng(_seed_for(seed, namespace, value))
    pattern = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
    return (pattern / np.sqrt(np.float32(dimension))).astype(np.float32)


@dataclass(frozen=True)
class Retrieval:
    value: str
    confidence_margin: float
    reconstructed_norm: float
    scores: dict[str, float]


class HolographicStateField:
    """Multi-channel image-like field for distributed key/value associations."""

    def __init__(
        self,
        value_vocabulary: Iterable[str],
        *,
        height: int = 32,
        width: int = 32,
        channels: int = 64,
        seed: int = 7,
    ) -> None:
        vocabulary = tuple(dict.fromkeys(str(v) for v in value_vocabulary))
        if not vocabulary:
            raise ValueError("value_vocabulary must contain at least one value")
        if min(height, width, channels) <= 0:
            raise ValueError("field dimensions must be positive")
        self.height, self.width, self.channels = int(height), int(width), int(channels)
        self.seed = int(seed)
        self.value_vocabulary = vocabulary
        self._spatial_size = self.height * self.width
        self._field: FloatArray = np.zeros(
            (self.height, self.width, self.channels), dtype=np.float32
        )
        self._value_codes: dict[str, FloatArray] = {
            value: _bipolar_pattern(self.seed, "value", value, self.channels)
            for value in vocabulary
        }
        self.writes = 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._field.shape

    @property
    def memory_bytes(self) -> int:
        """NumPy payload bytes for the field and fixed vocabulary codebook."""
        return int(self._field.nbytes + sum(v.nbytes for v in self._value_codes.values()))

    @property
    def field_energy(self) -> float:
        return float(np.linalg.norm(self._field))

    def _key_pattern(self, key: str) -> FloatArray:
        flat = _bipolar_pattern(self.seed, "key", str(key), self._spatial_size)
        return flat.reshape(self.height, self.width)

    def write(self, key: str, value: str, *, strength: float = 1.0) -> None:
        """Superpose an association; writing does not allocate a memory slot."""
        if value not in self._value_codes:
            raise KeyError(f"value {value!r} is not in the fixed vocabulary")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be a finite positive number")
        spatial = self._key_pattern(key)
        code = self._value_codes[value]
        self._field += np.float32(strength) * spatial[:, :, None] * code[None, None, :]
        self.writes += 1

    def write_pattern(self, pattern: FloatArray, value: str, *, strength: float = 1.0) -> None:
        """Write using an externally generated spatial cue (research interface).

        The caller owns cue generation; the field retains no per-record keys.
        Patterns must be finite, non-zero, and have the expected flattened size.
        """
        if value not in self._value_codes:
            raise KeyError(f"value {value!r} is not in the fixed vocabulary")
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError("strength must be a finite positive number")
        cue = np.asarray(pattern, dtype=np.float32).reshape(-1)
        if cue.size != self._spatial_size or not np.all(np.isfinite(cue)):
            raise ValueError("pattern must be finite and match the spatial dimension")
        norm = float(np.linalg.norm(cue))
        if norm <= 0:
            raise ValueError("pattern must have non-zero norm")
        spatial = (cue / np.float32(norm)).reshape(self.height, self.width)
        code = self._value_codes[value]
        self._field += np.float32(strength) * spatial[:, :, None] * code[None, None, :]
        self.writes += 1

    def retrieve_pattern(self, pattern: FloatArray) -> Retrieval:
        """Retrieve using an externally generated spatial cue."""
        cue = np.asarray(pattern, dtype=np.float32).reshape(-1)
        if cue.size != self._spatial_size or not np.all(np.isfinite(cue)):
            raise ValueError("pattern must be finite and match the spatial dimension")
        norm = float(np.linalg.norm(cue))
        if norm <= 0:
            raise ValueError("pattern must have non-zero norm")
        spatial = (cue / np.float32(norm)).reshape(self.height, self.width)
        reconstructed = np.einsum("hw,hwc->c", spatial, self._field, optimize=True)
        scores_array = np.asarray(
            [np.dot(reconstructed, self._value_codes[v]) for v in self.value_vocabulary],
            dtype=np.float32,
        )
        order = np.argsort(scores_array)
        best = int(order[-1])
        second = float(scores_array[order[-2]]) if len(order) > 1 else 0.0
        return Retrieval(
            value=self.value_vocabulary[best],
            confidence_margin=float(scores_array[best]) - second,
            reconstructed_norm=float(np.linalg.norm(reconstructed)),
            scores={
                value: float(score)
                for value, score in zip(self.value_vocabulary, scores_array, strict=True)
            },
        )

    def write_pattern_delta(self, pattern: FloatArray, value: str, *, rate: float = 1.0) -> float:
        """Error-correcting (delta-rule) write: store only what the field does not already predict.

            F <- F + rate * K (x) (V(value) - F.K)

        With a unit-norm cue and ``rate=1`` the field afterwards returns exactly ``V(value)`` for this
        cue, whatever it returned before. Consequences that the additive (Hebbian) write does not have:

        * supersession in place: rewriting a key with a new value replaces the old one without an
          external index and without knowing the old value;
        * idempotence: repeating a write changes nothing (no unbounded reinforcement);
        * decorrelation: for similar cues only the component not explained by earlier writes is stored.

        This is the classical Widrow-Hoff / Kaczmarz update of a correlation-matrix memory, not a new
        learning rule. One pass is exact for the *latest* write only: a later write with a
        non-orthogonal cue perturbs earlier associations by (cue overlap) x (its residual). At most
        H*W linearly independent cues can be satisfied simultaneously. Returns the residual norm
        (0 means the field already held this association).
        """
        if value not in self._value_codes:
            raise KeyError(f"value {value!r} is not in the fixed vocabulary")
        if not np.isfinite(rate) or not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        cue = np.asarray(pattern, dtype=np.float32).reshape(-1)
        if cue.size != self._spatial_size or not np.all(np.isfinite(cue)):
            raise ValueError("pattern must be finite and match the spatial dimension")
        norm = float(np.linalg.norm(cue))
        if norm <= 0:
            raise ValueError("pattern must have non-zero norm")
        spatial = (cue / np.float32(norm)).reshape(self.height, self.width)
        predicted = np.einsum("hw,hwc->c", spatial, self._field, optimize=True)
        residual = self._value_codes[value] - predicted
        self._field += np.float32(rate) * spatial[:, :, None] * residual[None, None, :]
        self.writes += 1
        return float(np.linalg.norm(residual))

    def write_delta(self, key: str, value: str, *, rate: float = 1.0) -> float:
        """Delta-rule write for a string key; see ``write_pattern_delta``."""
        return self.write_pattern_delta(self._key_pattern(key).reshape(-1), value, rate=rate)

    def reinforce(self, key: str, value: str, *, amount: float = 0.1) -> None:
        """Strengthen an association with another small Hebbian-like trace."""
        self.write(key, value, strength=amount)

    def decay(self, rate: float) -> None:
        """Uniformly decay traces that are not being reinforced."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError("decay rate must be between 0 and 1")
        self._field *= np.float32(1.0 - rate)

    def retrieve(self, key: str) -> Retrieval:
        """Reconstruct a value vector by matched-filter correlation."""
        spatial = self._key_pattern(key)
        reconstructed = np.einsum("hw,hwc->c", spatial, self._field, optimize=True)
        scores_array = np.asarray(
            [np.dot(reconstructed, self._value_codes[v]) for v in self.value_vocabulary],
            dtype=np.float32,
        )
        order = np.argsort(scores_array)
        best = int(order[-1])
        second = float(scores_array[order[-2]]) if len(order) > 1 else 0.0
        return Retrieval(
            value=self.value_vocabulary[best],
            confidence_margin=float(scores_array[best]) - second,
            reconstructed_norm=float(np.linalg.norm(reconstructed)),
            scores={
                value: float(score)
                for value, score in zip(self.value_vocabulary, scores_array, strict=True)
            },
        )

    def clear(self) -> None:
        self._field.fill(0.0)
        self.writes = 0
