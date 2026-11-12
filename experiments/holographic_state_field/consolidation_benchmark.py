"""Stage 4b: falsification of the Stage 4a result, and a brain-inspired pipeline.

Stage 4a found one HSF-specific result: correlated keys are recovered when the delta rule is replayed. Two ways
that could be an artefact, and one way to push it further:

1. **Equal replay for the baselines.** HDC/HRR get error-driven replay too (perceptron retraining: when a record
   is recalled wrongly, subtract the wrong binding and add the right one), same number of passes.
2. **Exact least squares.** ``hsf_whitened``: a recursive-least-squares style write. It keeps the inverse cue
   covariance ``P`` and writes the residual along ``P k``, so it interferes minimally with stored cues, single
   pass, no replay. ``P`` is (H*W)^2 floats and is *counted*: ``hdc_replay_big`` is a flat memory with the same
   total float budget (field + P).
3. **Pattern separation** (``*_sep``): before storage each cue goes through a fixed random expansion followed by
   k-winners-take-all, the textbook model of dentate-gyrus sparse coding. Similar inputs get much less similar
   codes. Applied to HSF and to HDC alike.

Brain analogy, stated as an analogy only: separation ~ dentate gyrus; error-driven write ~ prediction-error gated
plasticity; replay ~ hippocampal replay consolidating into cortex. Nothing here is a biological model.

Protocol as Stage 4a: write n correlated keys, rewrite half with a new value, score updated / untouched keys.
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
from .supersession_benchmark import _value

DEFAULT_SEEDS = (7, 17, 37, 73, 137)
DEFAULT_COUNTS = (32, 64, 128, 256)
DEFAULT_SIMILARITIES = (0.5, 0.9)
METHODS = ("hsf_delta_replay", "hdc_replay", "hrr_replay", "hsf_whitened", "hdc_replay_big",
           "hsf_sep_delta", "hsf_sep_delta_replay", "hdc_sep_replay", "hsf_sep_homeo", "hsf_sep_homeo_replay")
REPLAY_PASSES = 3
SPARSITY = 0.05
RIDGE = 1e-3
HOMEO_FLOOR = 0.02


class PatternSeparator:
    """Fixed random expansion + k-winners-take-all: sparse codes whose overlap falls faster than input similarity."""

    def __init__(self, in_dim: int, out_dim: int, seed: int, sparsity: float = SPARSITY) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([seed, in_dim, out_dim, 4242]))
        self.weights = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
        self.k = max(1, int(round(out_dim * sparsity)))

    def __call__(self, cue: np.ndarray) -> np.ndarray:
        drive = self.weights @ np.asarray(cue, dtype=np.float32).reshape(-1)
        code = np.zeros_like(drive)
        code[np.argpartition(drive, -self.k)[-self.k:]] = 1.0
        return code / np.sqrt(np.float32(self.k))


class _HSF:
    def __init__(self, seed, cues, h, w, c, mode: str, separator=None) -> None:
        self.f = HolographicStateField(VOCABULARY, height=h, width=w, channels=c, seed=seed)
        self.cues = {n: (separator(v) if separator else v) for n, v in cues.items()}
        self.mode = mode
        n = h * w
        self.extra_floats = n * n if mode == "whitened" else 0
        self.P = np.eye(n, dtype=np.float64) / RIDGE if mode == "whitened" else None
        # "homeo": per-unit usage counts (N extra floats). Units that are active for many cues learn less, so a
        # write lands mostly on the units that distinguish this cue - a diagonal stand-in for P.
        self.homeo = mode.startswith("homeo")
        self.usage = np.zeros(n, dtype=np.float64)
        if self.homeo:
            self.extra_floats = n

    def write(self, name, value):
        cue = self.cues[name]
        if self.homeo:
            k = (cue / np.linalg.norm(cue)).astype(np.float64)
            self.usage += k * k
            direction = k / (HOMEO_FLOOR + self.usage)
            field = self.f._field.reshape(-1, self.f.channels)
            residual = self.f._value_codes[value].astype(np.float64) - k @ field
            field += np.outer(direction / (k @ direction), residual).astype(np.float32)
            self.f.writes += 1
            return
        if self.mode != "whitened":
            self.f.write_pattern_delta(cue, value)
            return
        k = (cue / np.linalg.norm(cue)).astype(np.float64)
        Pk = self.P @ k
        self.P -= np.outer(Pk, Pk) / (1.0 + k @ Pk)            # Sherman-Morrison: P <- (P^-1 + k k^T)^-1
        Pk = self.P @ k
        field = self.f._field.reshape(-1, self.f.channels)
        residual = self.f._value_codes[value].astype(np.float64) - k @ field
        field += (np.outer(Pk / (k @ Pk), residual)).astype(np.float32)   # exact for k, minimal for stored cues
        self.f.writes += 1

    def read(self, name):
        return self.f.retrieve_pattern(self.cues[name]).value

    def replay(self, final):
        if self.mode == "replay":
            for _ in range(REPLAY_PASSES):
                for name, value in final.items():
                    self.f.write_pattern_delta(self.cues[name], value)
        if self.mode == "homeo_replay":
            frozen = self.usage.copy()
            for _ in range(REPLAY_PASSES):
                for name, value in final.items():
                    self.write(name, value)
                    self.usage[:] = frozen                 # replay re-uses the learned usage; it does not recount


class _Flat:
    """HDC/HRR with error-driven replay (perceptron retraining)."""

    extra_floats = 0

    def __init__(self, store, cues, separator=None) -> None:
        self.store = store
        self.cues = {n: (separator(v) if separator else v) for n, v in cues.items()}

    def _key(self, name):
        cue = self.cues[name]
        return (cue / np.linalg.norm(cue)).astype(np.float32)

    def _correct(self, name, value):
        got = self.store.retrieve_vector(self.cues[name]).value
        if got != value:
            key = self._key(name)
            self.store._memory += self.store._bind(key, self.store._code("value", value)) \
                - self.store._bind(key, self.store._code("value", got))

    def write(self, name, value):
        if self.store.writes:
            self._correct(name, value)
            self.store.writes += 1
        else:
            self.store.write_vector(self.cues[name], value)

    def read(self, name):
        return self.store.retrieve_vector(self.cues[name]).value

    def replay(self, final):
        for _ in range(REPLAY_PASSES):
            for name, value in final.items():
                self._correct(name, value)


def _build(method: str, seed: int, count: int, rho: float, h: int, w: int, c: int):
    n_key, flat = h * w, h * w * c
    if method.startswith("hsf"):
        _, cues, _ = correlated_cues(seed, n_key, count, rho)
        sep = PatternSeparator(n_key, n_key, seed) if "_sep" in method else None
        mode = "whitened" if method == "hsf_whitened" else ("replay" if method.endswith("replay") else "delta")
        if "homeo" in method:
            mode = "homeo_replay" if method.endswith("replay") else "homeo"
        mem = _HSF(seed, cues, h, w, c, mode, sep)
        return mem, flat + mem.extra_floats
    dim = flat + n_key * n_key if method.endswith("_big") else flat
    _, cues, _ = correlated_cues(seed, dim, count, rho)
    sep = PatternSeparator(dim, dim, seed) if "_sep" in method else None
    cls = HRRMemory if method.startswith("hrr") else HDCMemory
    return _Flat(cls(VOCABULARY, dimension=dim, seed=seed), cues, sep), dim


def code_overlap(seed: int, dim: int, count: int, rho: float) -> dict:
    """Mean within-family cosine before and after pattern separation."""
    names, cues, realised = correlated_cues(seed, dim, count, rho)
    sep = PatternSeparator(dim, dim, seed)
    codes = {n: sep(v) for n, v in cues.items()}
    fam = [n for n in names if n.startswith("FAMILY_0:")]
    after = [float(codes[a] @ codes[b]) for i, a in enumerate(fam) for b in fam[i + 1:]]
    return {"input_cosine": realised, "separated_cosine": statistics.fmean(after) if after else None}


def run_one(method, seed, count, rho, h, w, c, update_fraction=0.5) -> dict:
    mem, floats = _build(method, seed, count, rho, h, w, c)
    names = [f"FAMILY_{i % FAMILIES}:ITEM_{i:05d}" for i in range(count)]
    original = {n: _value(i, seed) for i, n in enumerate(names)}
    rng = np.random.default_rng(np.random.SeedSequence([seed, count, 991]))
    upd_idx = sorted(rng.choice(count, size=max(1, int(round(count * update_fraction))), replace=False).tolist())
    final = dict(original)
    for n in names:
        mem.write(n, original[n])
    for i in upd_idx:
        final[names[i]] = _value(i, seed, shift=1 + (i % (len(VOCABULARY) - 1)))
        mem.write(names[i], final[names[i]])
    mem.replay(final)
    upd = {names[i] for i in upd_idx}
    got = {n: mem.read(n) for n in names}
    keep = [n for n in names if n not in upd]
    return {"method": method, "seed": seed, "count": count, "rho": rho, "floats": floats,
            "updated_accuracy": sum(got[n] == final[n] for n in upd) / len(upd),
            "untouched_accuracy": sum(got[n] == final[n] for n in keep) / len(keep) if keep else None,
            "overall_accuracy": sum(got[n] == final[n] for n in names) / count}


def run(*, seeds=DEFAULT_SEEDS, counts=DEFAULT_COUNTS, similarities=DEFAULT_SIMILARITIES,
        height=16, width=16, channels=16, methods=METHODS) -> dict:
    if not seeds or not counts or not similarities:
        raise ValueError("seeds, counts, and similarities must be non-empty")
    runs = [run_one(m, s, n, rho, height, width, channels)
            for rho in similarities for n in counts for s in seeds for m in methods]
    agg = []
    for rho in similarities:
        for n in counts:
            for m in methods:
                rows = [r for r in runs if (r["method"], r["count"], r["rho"]) == (m, n, rho)]
                agg.append({"method": m, "count": n, "rho": rho, "floats": rows[0]["floats"],
                            "updated_accuracy_mean": statistics.fmean(r["updated_accuracy"] for r in rows),
                            "untouched_accuracy_mean": statistics.fmean(r["untouched_accuracy"] for r in rows),
                            "overall_accuracy_mean": statistics.fmean(r["overall_accuracy"] for r in rows),
                            "overall_accuracy_min": min(r["overall_accuracy"] for r in rows)})
    overlap = {str(rho): code_overlap(seeds[0], height * width, 64, rho) for rho in similarities}
    return {"experiment": "hsf_consolidation_v1",
            "config": {"seeds": list(seeds), "counts": list(counts), "similarities": list(similarities),
                       "shape": [height, width, channels], "replay_passes": REPLAY_PASSES, "sparsity": SPARSITY, "ridge": RIDGE},
            "boundary": "Closed 8-value vocabulary, synthetic bipolar cues. Replay re-presents every record. "
                        "Random projections and codebooks are regenerated from seeds and not counted; P is counted.",
            "pattern_separation_overlap": overlap, "aggregates": agg, "runs": runs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/hsf_consolidation_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("pattern separation (within-family cosine):", result["pattern_separation_overlap"])
    for rho in result["config"]["similarities"]:
        print(f"rho={rho}  overall accuracy mean (floats)")
        for n in result["config"]["counts"]:
            cells = [a for a in result["aggregates"] if a["rho"] == rho and a["count"] == n]
            print(f"  n={n:<4}" + "  ".join(f"{a['method']}: {a['overall_accuracy_mean']:.2f}" for a in cells))
    print({a["method"]: a["floats"] for a in result["aggregates"][:len(METHODS)]})


if __name__ == "__main__":
    main()
