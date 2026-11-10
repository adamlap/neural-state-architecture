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
