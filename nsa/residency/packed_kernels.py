"""Reference packed INT/ternary kernels used by hardware backends.

These implementations are correctness references. Accelerator backends can
replace them without changing residency or scheduling interfaces.
"""

from __future__ import annotations

from typing import Sequence


def unpack_unsigned(values: bytes, bits: int, count: int) -> list[int]:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    if count < 0:
        raise ValueError("count must be non-negative")
    mask = (1 << bits) - 1
    result: list[int] = []
    accumulator = 0
    available = 0
    iterator = iter(values)
    while len(result) < count:
        while available < bits:
            try:
                accumulator |= next(iterator) << available
            except StopIteration as exc:
                raise ValueError("packed payload is shorter than requested") from exc
            available += 8
        result.append(accumulator & mask)
        accumulator >>= bits
        available -= bits
    return result


def unpack_signed(values: bytes, bits: int, count: int) -> list[int]:
    unsigned = unpack_unsigned(values, bits, count)
    midpoint = 1 << (bits - 1)
    modulus = 1 << bits
    return [value - modulus if value >= midpoint else value for value in unsigned]


def unpack_ternary(values: bytes, count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    result: list[int] = []
    for byte in values:
        value = byte
        for _ in range(5):
            if len(result) == count:
                return result
            result.append((value % 3) - 1)
            value //= 3
    if len(result) != count:
        raise ValueError("packed ternary payload is shorter than requested")
    return result


def int4_linear_reference(
    weights: Sequence[int],
    inputs: Sequence[float],
    scale: float,
    zero_point: int = 0,
) -> float:
    if len(weights) != len(inputs):
        raise ValueError("weights and inputs must have equal length")
    return sum((weight - zero_point) * scale * value for weight, value in zip(weights, inputs))
