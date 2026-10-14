"""Small, dependency-free statistics for paired benchmark comparisons."""
from __future__ import annotations

import math
from math import comb
from typing import Iterable, Mapping


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - a) / d), min(1.0, (c + a) / d))


def mcnemar_exact(a: Mapping, b: Mapping) -> dict:
    """Exact two-sided McNemar test on paired boolean outcomes keyed by episode."""
    keys = a.keys() & b.keys()
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n)
    return {"paired": len(keys), "only_first": only_a, "only_second": only_b, "p": p}


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    ps = list(pvalues)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted = [0.0] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted
