"""Stage 4: conflicting updates (supersession) and error-correcting writes.

Question: when a key is rewritten with a new value, does the memory return the new value, without an
external index of what was stored?

Update policies compared at an equal float budget (HSF H x W x C vs flat vectors of dimension H*W*C):

* ``*_additive``  - the plain superposition write of each method (old and new traces coexist).
* ``*_cleanup``   - index-free "retrieve, clean up to the nearest vocabulary code, subtract it, add the
                    new one"; applied only when the retrieved margin exceeds ``CLEANUP_MARGIN`` times the margin of
                    one clean association (methods scale scores differently). This is
                    the strongest update a flat superposition can do by itself, so it is the fair baseline.
* ``hsf_delta``   - delta-rule write (``HolographicStateField.write_pattern_delta``), one pass.
* ``hsf_delta_replay`` - the same, with the final record set presented ``REPLAY_PASSES`` times. Replay needs
                    the records again, so it is *not* index-free; it bounds what the rule can reach.

Protocol: write ``count`` keys (within-family cosine ``rho``), then rewrite ``update_fraction`` of them
with a different value. Report accuracy on updated keys (the latest value), the stale rate (the old value
is returned), and accuracy on untouched keys (collateral damage).

Why the delta rule is HSF-specific: for elementwise bipolar binding (HDC) the residual write
``m + k*(v - k*m)`` equals ``k*v`` - it erases every other association. A matrix memory has a null space
per cue; a flat bound vector does not.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .baselines import HDCMemory, HRRMemory
from .correlated_key_benchmark import FAMILIES, VOCABULARY, correlated_cues
from .field import HolographicStateField

DEFAULT_SEEDS = (7, 17, 37, 73, 137)
DEFAULT_COUNTS = (16, 32, 64, 128, 256)
DEFAULT_SIMILARITIES = (0.0, 0.5, 0.9)
METHODS = ("hsf_additive", "hsf_cleanup", "hsf_delta", "hsf_delta_replay", "hdc_additive", "hdc_cleanup", "hrr_additive", "hrr_cleanup")
CLEANUP_MARGIN = 0.25
REPLAY_PASSES = 3


def _value(i: int, seed: int, shift: int = 0) -> str:
    return VOCABULARY[((i // FAMILIES) * 3 + (i % FAMILIES) * 5 + seed + shift) % len(VOCABULARY)]


class _Vec:
    """Adapter over HDC/HRR with additive and cleanup-subtract updates."""

    def __init__(self, store, cues, cleanup: bool) -> None:
        self.store, self.cues, self.cleanup = store, cues, cleanup
        # Methods scale their scores differently (HDC returns 1/sqrt(D) for a clean item), so the cleanup
        # threshold is relative to the margin of a single clean association in an empty memory.
        probe = type(store)(store.vocabulary, dimension=store.dimension, seed=store.seed)
        probe.write("__unit__", store.vocabulary[0])
        self.unit = probe.retrieve("__unit__").margin

    def write(self, name: str, value: str) -> None:
        if self.cleanup and self.store.writes:
            got = self.store.retrieve_vector(self.cues[name])
            if got.value != value and got.margin > CLEANUP_MARGIN * self.unit:
                key = self.cues[name] / np.linalg.norm(self.cues[name])
                self.store._memory -= self.store._bind(key.astype(np.float32), self.store._code("value", got.value))
        self.store.write_vector(self.cues[name], value)

    def read(self, name: str) -> str:
        return self.store.retrieve_vector(self.cues[name]).value

    def replay(self, final: dict) -> None:
        pass


class _Field:
    def __init__(self, store: HolographicStateField, cues, mode: str) -> None:
        self.store, self.cues, self.mode = store, cues, mode

    def write(self, name: str, value: str) -> None:
        if self.mode.startswith("delta"):
            self.store.write_pattern_delta(self.cues[name], value)
            return
        if self.mode == "cleanup" and self.store.writes:
            got = self.store.retrieve_pattern(self.cues[name])
            if got.value != value and got.confidence_margin > CLEANUP_MARGIN:
                cue = (self.cues[name] / np.linalg.norm(self.cues[name])).reshape(self.store.height, self.store.width)
                self.store._field -= cue[:, :, None] * self.store._value_codes[got.value][None, None, :]
        self.store.write_pattern(self.cues[name], value)

    def read(self, name: str) -> str:
        return self.store.retrieve_pattern(self.cues[name]).value

    def replay(self, final: dict) -> None:
        if self.mode == "delta_replay":
            for _ in range(REPLAY_PASSES):
                for name, value in final.items():
                    self.store.write_pattern_delta(self.cues[name], value)


def _build(method: str, seed: int, count: int, rho: float, h: int, w: int, c: int):
    family, mode = method.split("_", 1)
    if family == "hsf":
        _, cues, realised = correlated_cues(seed, h * w, count, rho)
        return _Field(HolographicStateField(VOCABULARY, height=h, width=w, channels=c, seed=seed), cues, mode), realised
    _, cues, realised = correlated_cues(seed, h * w * c, count, rho)
    cls = HDCMemory if family == "hdc" else HRRMemory
    return _Vec(cls(VOCABULARY, dimension=h * w * c, seed=seed), cues, mode == "cleanup"), realised


def run_one(method: str, seed: int, count: int, rho: float, update_fraction: float, h: int, w: int, c: int) -> dict:
    mem, realised = _build(method, seed, count, rho, h, w, c)
    names = [f"FAMILY_{i % FAMILIES}:ITEM_{i:05d}" for i in range(count)]
    original = {n: _value(i, seed) for i, n in enumerate(names)}
    rng = np.random.default_rng(np.random.SeedSequence([seed, count, 991]))
    updated_idx = sorted(rng.choice(count, size=max(1, int(round(count * update_fraction))), replace=False).tolist())
    final = dict(original)
    for n in names:
        mem.write(n, original[n])
    for i in updated_idx:
        final[names[i]] = _value(i, seed, shift=1 + (i % (len(VOCABULARY) - 1)))
        mem.write(names[i], final[names[i]])
    mem.replay(final)
    upd = {names[i] for i in updated_idx}
    got = {n: mem.read(n) for n in names}
    n_upd, n_keep = len(upd), count - len(upd)
    return {
        "method": method, "seed": seed, "count": count, "rho": rho, "realised_cosine": realised,
        "updated_accuracy": sum(got[n] == final[n] for n in upd) / n_upd,
        "stale_rate": sum(got[n] == original[n] for n in upd) / n_upd,
        "untouched_accuracy": (sum(got[n] == final[n] for n in names if n not in upd) / n_keep) if n_keep else None,
        "overall_accuracy": sum(got[n] == final[n] for n in names) / count,
    }


def run(*, seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS, similarities=DEFAULT_SIMILARITIES, update_fraction: float = 0.5,
        height: int = 16, width: int = 16, channels: int = 16, methods=METHODS) -> dict:
    if not seeds or not counts or not similarities:
        raise ValueError("seeds, counts, and similarities must be non-empty")
    if not 0.0 < update_fraction <= 1.0:
        raise ValueError("update_fraction must be in (0, 1]")
    if channels < len(VOCABULARY):
        raise ValueError("channels must be at least the vocabulary size")
    runs = [run_one(m, s, n, rho, update_fraction, height, width, channels)
            for rho in similarities for n in counts for s in seeds for m in methods]
    aggregates = []
    for rho in similarities:
        for n in counts:
            for m in methods:
                rows = [r for r in runs if r["method"] == m and r["count"] == n and r["rho"] == rho]
                keep = [r["untouched_accuracy"] for r in rows if r["untouched_accuracy"] is not None]
                aggregates.append({
                    "method": m, "count": n, "rho": rho,
                    "updated_accuracy_mean": statistics.fmean(r["updated_accuracy"] for r in rows),
                    "updated_accuracy_min": min(r["updated_accuracy"] for r in rows),
                    "stale_rate_mean": statistics.fmean(r["stale_rate"] for r in rows),
                    "untouched_accuracy_mean": statistics.fmean(keep) if keep else None,
                    "overall_accuracy_mean": statistics.fmean(r["overall_accuracy"] for r in rows),
                })
    return {
        "experiment": "hsf_supersession_v1",
        "config": {"seeds": list(seeds), "counts": list(counts), "similarities": list(similarities),
                   "update_fraction": update_fraction, "shape": [height, width, channels],
                   "flat_dimension": height * width * channels, "key_dimension_hsf": height * width,
                   "cleanup_margin": CLEANUP_MARGIN, "replay_passes": REPLAY_PASSES, "vocabulary": len(VOCABULARY)},
        "boundary": ("Closed 8-value vocabulary, synthetic bipolar cues, equal float payload but unequal compute. "
                     "hsf_delta_replay re-presents the records and is not index-free. No LLM, no semantics."),
        "aggregates": aggregates, "runs": runs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/hsf_supersession_v1.json")
    p.add_argument("--update-fraction", type=float, default=0.5)
    args = p.parse_args()
    result = run(update_fraction=args.update_fraction)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for rho in result["config"]["similarities"]:
        print(f"rho={rho}  (updated acc / stale / untouched acc)")
        for n in result["config"]["counts"]:
            cells = [a for a in result["aggregates"] if a["rho"] == rho and a["count"] == n]
            print(f"  n={n:<4}" + "  ".join(
                f"{a['method']}: {a['updated_accuracy_mean']:.2f}/{a['stale_rate_mean']:.2f}/{a['untouched_accuracy_mean']:.2f}" for a in cells))


if __name__ == "__main__":
    main()
