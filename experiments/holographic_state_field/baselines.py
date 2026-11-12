"""Small, deterministic associative-memory baselines for HSF comparisons."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

def _seed(seed: int, namespace: str, key: str) -> int:
    raw = hashlib.sha256(f"{seed}\0{namespace}\0{key}".encode()).digest()
    return int.from_bytes(raw[:8], "little")

def normalised_bipolar(seed: int, namespace: str, key: str, dimension: int) -> FloatArray:
    rng = np.random.default_rng(_seed(seed, namespace, key))
    vector = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dimension)
    return vector / np.sqrt(np.float32(dimension))

@dataclass(frozen=True)
class BaselineRetrieval:
    value: str
    margin: float

class _VectorSuperposition:
    def __init__(self, vocabulary: Iterable[str], *, dimension: int, seed: int) -> None:
        self.vocabulary = tuple(dict.fromkeys(str(v) for v in vocabulary))
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension, self.seed = int(dimension), int(seed)
        self._memory = np.zeros(self.dimension, dtype=np.float32)
        self.writes = 0

    def _code(self, namespace: str, value: str) -> FloatArray:
        return normalised_bipolar(self.seed, namespace, value, self.dimension)

    def _key(self, key: str) -> FloatArray:
        return self._code("key", str(key))

    @property
    def memory_bytes(self) -> int:
        # Key/value codes are deterministic and regenerated; only persistent state is counted.
        return int(self._memory.nbytes)

    def _bind(self, key: FloatArray, value: FloatArray) -> FloatArray:
        raise NotImplementedError

    def _unbind(self, memory: FloatArray, key: FloatArray) -> FloatArray:
        raise NotImplementedError

    def write(self, key: str, value: str) -> None:
        if value not in self.vocabulary:
            raise KeyError(value)
        self._memory += self._bind(self._key(key), self._code("value", value))
        self.writes += 1

    def write_vector(self, key_vector: FloatArray, value: str) -> None:
        """Write a supplied key vector for controlled representation experiments."""
        if value not in self.vocabulary:
            raise KeyError(value)
        key = np.asarray(key_vector, dtype=np.float32).reshape(-1)
        if key.size != self.dimension or not np.all(np.isfinite(key)):
            raise ValueError("key vector must be finite and match the memory dimension")
        norm = float(np.linalg.norm(key))
        if norm <= 0:
            raise ValueError("key vector must have non-zero norm")
        self._memory += self._bind(key / np.float32(norm), self._code("value", value))
        self.writes += 1

    def retrieve(self, key: str) -> BaselineRetrieval:
        return self.retrieve_vector(self._key(key))

    def retrieve_vector(self, key_vector: FloatArray) -> BaselineRetrieval:
        """Retrieve using a supplied key vector."""
        key = np.asarray(key_vector, dtype=np.float32).reshape(-1)
        if key.size != self.dimension or not np.all(np.isfinite(key)):
            raise ValueError("key vector must be finite and match the memory dimension")
        norm = float(np.linalg.norm(key))
        if norm <= 0:
            raise ValueError("key vector must have non-zero norm")
        vector = self._unbind(self._memory, key / np.float32(norm))
        scores = np.asarray([
            np.dot(vector, self._code("value", v)) for v in self.vocabulary
        ], dtype=np.float32)
        order = np.argsort(scores)
        best = int(order[-1])
        second = float(scores[order[-2]]) if len(order) > 1 else 0.0
        return BaselineRetrieval(self.vocabulary[best], float(scores[best]) - second)

class HRRMemory(_VectorSuperposition):
    """Holographic reduced representations with circular-convolution binding."""
    def _bind(self, key: FloatArray, value: FloatArray) -> FloatArray:
        n = self.dimension
        return (np.fft.irfft(np.fft.rfft(key) * np.fft.rfft(value), n=n)
                * np.sqrt(np.float32(n))).astype(np.float32)

    def _unbind(self, memory: FloatArray, key: FloatArray) -> FloatArray:
        n = self.dimension
        return (np.fft.irfft(np.fft.rfft(memory) * np.conjugate(np.fft.rfft(key)), n=n)
                * np.sqrt(np.float32(n))).astype(np.float32)

class HDCMemory(_VectorSuperposition):
    """Bipolar hyperdimensional binding and additive bundling."""
    def _bind(self, key: FloatArray, value: FloatArray) -> FloatArray:
        return (key * value * np.sqrt(np.float32(self.dimension))).astype(np.float32)

    def _unbind(self, memory: FloatArray, key: FloatArray) -> FloatArray:
        return (memory * key).astype(np.float32)

class ExactDictionary:
    """Perfect-recall control with explicit key/value associations."""
    def __init__(self) -> None:
        self._records: dict[str, str] = {}

    @property
    def memory_bytes(self) -> int:
        # Python strings/object overhead is implementation-dependent; measured separately.
        return 0

    @property
    def record_count(self) -> int:
        return len(self._records)

    def write(self, key: str, value: str) -> None:
        self._records[str(key)] = str(value)

    def retrieve(self, key: str) -> BaselineRetrieval:
        return BaselineRetrieval(self._records[str(key)], float("inf"))
