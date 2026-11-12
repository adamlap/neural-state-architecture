"""Route an extracted fact against what memory already holds: write, ignore, or hold for confirmation.

A familiarity score cannot tell a typo from a real change (both are "same line, different value"). The memory can: it knows the current value
for the key. Rules, applied to a trusted-or-untrusted observation alike (the integrity gate in ``TemporalMemoryStore`` still decides what an
untrusted write may do):

* unknown key                                   -> ``write``
* same value as the current one                 -> ``ignore`` (a reminder; no new version)
* different value, explicit update cue          -> ``write`` (supersede; the author said it changed)
* different value, a *near-duplicate* of the current one (small edit distance) and no update cue
                                                -> ``hold``: probably transcription noise. It becomes a write only if the same value is observed
                                                   again (random typos do not repeat; a genuine change that is restated does)
* different value, not a near-duplicate         -> ``write``

Limit: a genuine change to a close value stated without an update cue is delayed until it is mentioned twice.
"""
from __future__ import annotations

from typing import Literal

Route = Literal["write", "ignore", "hold"]


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, early-exiting above ``cap`` (returns cap + 1)."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def is_near_duplicate(new: str, current: str, *, max_edits: int = 1, min_length: int = 4) -> bool:
    new, current = new.lower(), current.lower()
    return min(len(new), len(current)) >= min_length and edit_distance(new, current, max_edits) <= max_edits


def route_fact(current_value: str | None, new_value: str, kind: str, *, max_edits: int = 1) -> Route:
    if current_value is None:
        return "write"
    if new_value.lower() == current_value.lower():
        return "ignore"
    if kind == "update":
        return "write"
    return "hold" if is_near_duplicate(new_value, current_value, max_edits=max_edits) else "write"


__all__ = ["route_fact", "is_near_duplicate", "edit_distance", "Route"]
