"""Stage 5a: do NSA's real fact keys behave like the correlated families of Stage 4, and does the 4h recipe still win?

Keys are the normalised "entity attribute" phrases NSA's extractor produces (e.g. "admin billing email"). A cue is the normalised
sum of fixed random bipolar codes of the key's *tokens*, so keys that share a word share a component: correlation comes from the
data, not from a dial. Values are opaque strings drawn from an open vocabulary (each value gets a hashed short code; decoding
matches against the set of value strings seen, which any deployment must keep).

Task: write every key, rewrite half with a new value, replay, then read every key. Methods: the exact dictionary (the thing NSA
has today, perfect recall by construction), HSF (additive / delta / separation+homeostatic+replay) and HDC with separation and
replay, at equal float budgets for the three approximate memories. The exact dictionary's cost is reported separately in floats
only as an upper reference (n keys x (cue dim + code dim)); a real dictionary also stores key and value strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from random import Random

import numpy as np

from . import consolidation_benchmark as cb
from .baselines import HDCMemory
from .field import HolographicStateField

ENTITIES = ("staging", "billing", "payments", "auth", "search", "reports", "gateway", "mailer", "scheduler", "uploads")
ATTRS = ("database port", "admin email", "api region", "timeout seconds", "cache size", "queue name", "log level", "owner team")
EVAL_SEEDS = (11, 23, 47, 59, 71, 83, 97, 101, 113, 127)
METHODS = ("exact", "hsf_additive", "hsf_sep_delta", "hsf_sep_homeo_replay", "hdc_sep_replay")


def key_name(entity: str, attribute: str) -> str:
    from nsa.memory.extract import normalize_key
    return normalize_key(f"{entity} {attribute}")


def token_cue(key: str, dim: int, seed: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in key.split():
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}|{tok}".encode()).digest()[:8], "little"))
        vec += rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=dim)
    return vec / np.linalg.norm(vec)


def episode(seed: int, n_keys: int):
    rng = Random(seed)
    combos = [(e, a) for e in ENTITIES for a in ATTRS]
    rng.shuffle(combos)
    keys = [key_name(e, a) for e, a in combos[:n_keys]]
    token = lambda: "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8))
    original = {k: token() for k in keys}
    updated = rng.sample(keys, k=n_keys // 2)
    final = dict(original)
    for k in updated:
        final[k] = token()
    return keys, original, final, updated


def mean_cosine(cues: dict[str, np.ndarray], keys: list[str]) -> float:
    vals = [float(cues[a] @ cues[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
    return statistics.fmean(vals)


def run_one(method: str, seed: int, n_keys: int, h: int, w: int, c: int, sparsity: float = 0.02) -> dict:
    keys, original, final, updated = episode(seed, n_keys)
    if method == "exact":
        got = dict(final)
        floats = n_keys * (h * w + c)
        key_cos = sep_cos = None
    else:
        old = cb.SPARSITY
        cb.SPARSITY = sparsity
        try:
            vocab = tuple(sorted({*original.values(), *final.values()}))                 # the open-vocabulary value store
            key_dim = h * w if method.startswith("hsf") else h * w * c
            cues = {k: token_cue(k, key_dim, seed) for k in keys}
            key_cos = mean_cosine(cues, keys)
            sep = cb.PatternSeparator(key_dim, key_dim, seed) if "_sep" in method else None
            if sep:
                sep_cos = mean_cosine({k: sep(v) for k, v in cues.items()}, keys)
            else:
                sep_cos = None
            if method.startswith("hsf"):
                mode = {"hsf_additive": "delta", "hsf_sep_delta": "delta", "hsf_sep_homeo_replay": "homeo_replay"}[method]
                mem = cb._HSF(seed, cues, h, w, c, mode, sep, vocab)
                floats = h * w * c + (h * w if "homeo" in mode else 0)
                if method == "hsf_additive":
                    mem.write = lambda name, value, _m=mem: _m.f.write_pattern(_m.cues[name], value)
            else:
                mem = cb._Flat(HDCMemory(vocab, dimension=key_dim, seed=seed), cues, sep)
                floats = key_dim
            for k in keys:
                mem.write(k, original[k])
            for k in updated:
                mem.write(k, final[k])
            if method.endswith("replay"):
                mem.replay(final)
            got = {k: mem.read(k) for k in keys}
        finally:
            cb.SPARSITY = old
    acc = sum(got[k] == final[k] for k in keys) / n_keys
    upd = sum(got[k] == final[k] for k in updated) / len(updated)
    return {"method": method, "seed": seed, "n_keys": n_keys, "floats": floats, "accuracy": acc, "updated_accuracy": upd,
            "key_cosine": key_cos, "separated_cosine": sep_cos}


def run(*, seeds=EVAL_SEEDS, key_counts=(40, 80), shapes=((16, 16, 16), (32, 32, 16)), methods=METHODS) -> dict:
    if not seeds or not key_counts:
        raise ValueError("seeds and key_counts must be non-empty")
    rows = []
    for n in key_counts:
        for sh in shapes:
            for m in methods:
                runs = [run_one(m, s, n, *sh) for s in seeds]
                cos = [r["key_cosine"] for r in runs if r["key_cosine"] is not None]
                sep = [r["separated_cosine"] for r in runs if r["separated_cosine"] is not None]
                rows.append({"n_keys": n, "shape": list(sh), "method": m, "floats": runs[0]["floats"],
                             "accuracy_mean": statistics.fmean(r["accuracy"] for r in runs), "accuracy_min": min(r["accuracy"] for r in runs),
                             "updated_accuracy_mean": statistics.fmean(r["updated_accuracy"] for r in runs),
                             "mean_key_cosine": statistics.fmean(cos) if cos else None, "mean_separated_cosine": statistics.fmean(sep) if sep else None})
    return {"experiment": "hsf_nsa_keys_v1", "seeds": list(seeds), "rows": rows,
            "boundary": "NSA-style 'entity attribute' keys from a 10x8 grid, token-sum cues, opaque string values, open-vocabulary value store; "
                        "the exact dictionary is perfect by construction and is not budget-matched."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/hsf_nsa_keys_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in result["rows"]:
        cos = f" key cos {r['mean_key_cosine']:.2f}->{r['mean_separated_cosine']:.2f}" if r["mean_separated_cosine"] is not None else (
            f" key cos {r['mean_key_cosine']:.2f}" if r["mean_key_cosine"] is not None else "")
        print(f"n={r['n_keys']} {r['shape']} {r['method']:<22} floats={r['floats']:<6} acc {r['accuracy_mean']:.2f} (min {r['accuracy_min']:.2f}) upd {r['updated_accuracy_mean']:.2f}{cos}")


if __name__ == "__main__":
    main()
