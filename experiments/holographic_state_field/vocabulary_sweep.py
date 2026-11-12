"""Stage 4f: larger value vocabularies at equal float budgets.

HSF stores a value as a C-dimensional code, so a vocabulary of V values needs C >= V channels (bipolar random codes
stay near-orthogonal only while C is large relative to V), which takes floats away from the key plane H x W and
therefore from capacity. A flat HDC vector has no such split. This sweep asks how much that costs.

For each vocabulary size V and field budget B (floats), HSF candidates use C in {V, 2V} (C >= V) and a roughly
square key plane with H*W = B/C; HDC uses D = B. Sparsity (2 % or 5 %) is chosen per method on selection seeds and
evaluated on disjoint seeds. Cosine 0.9, full replay, n records with half rewritten.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from . import consolidation_benchmark as cb

SELECT_SEEDS = (1, 2, 3, 4, 5)
EVAL_SEEDS = (11, 23, 47, 59, 71, 83, 97, 101, 113, 127)
HSF, HDC = "hsf_sep_homeo_replay", "hdc_sep_replay"


def plane(cells: int) -> tuple[int, int]:
    h = 2 ** int(math.log2(math.isqrt(cells)))
    return h, cells // h


def shapes_for(vocab: int, budget: int) -> list[tuple[int, int, int]]:
    out = []
    for c in (vocab, 2 * vocab):
        if budget % c == 0 and budget // c >= 16:
            h, w = plane(budget // c)
            out.append((h, w, c))
    return out


def _acc(method, seeds, n, rho, shape, sparsity, vocab):
    old = cb.SPARSITY
    cb.SPARSITY = sparsity
    try:
        return [cb.run_one(method, s, n, rho, *shape, vocab_size=vocab)["overall_accuracy"] for s in seeds]
    finally:
        cb.SPARSITY = old


def run(*, vocabs=(8, 32, 128), budgets=(4096, 16384), counts=(32, 64), rho=0.9, sparsities=(0.02, 0.05),
        select_seeds=SELECT_SEEDS, eval_seeds=EVAL_SEEDS) -> dict:
    if set(select_seeds) & set(eval_seeds):
        raise ValueError("selection and evaluation seeds must be disjoint")
    rows = []
    for vocab in vocabs:
        for budget in budgets:
            shapes = shapes_for(vocab, budget)
            if not shapes:
                continue
            for n in counts:
                hsf_grid = {(sh, sp): statistics.fmean(_acc(HSF, select_seeds, n, rho, sh, sp, vocab)) for sh in shapes for sp in sparsities}
                (shape, hsp), _ = max(hsf_grid.items(), key=lambda kv: kv[1])
                flat_shape = (1, budget, 1)                                   # D = budget; run_one only uses the product
                hdc_grid = {sp: statistics.fmean(_acc(HDC, select_seeds, n, rho, flat_shape, sp, vocab)) for sp in sparsities}
                dsp = max(hdc_grid, key=hdc_grid.get)
                hsf_eval = _acc(HSF, eval_seeds, n, rho, shape, hsp, vocab)
                hdc_eval = _acc(HDC, eval_seeds, n, rho, flat_shape, dsp, vocab)
                rows.append({"vocab": vocab, "budget": budget, "count": n, "hsf_shape": list(shape), "key_dim": shape[0] * shape[1],
                             "hsf_sparsity": hsp, "hdc_sparsity": dsp,
                             "hsf_mean": statistics.fmean(hsf_eval), "hsf_min": min(hsf_eval),
                             "hdc_mean": statistics.fmean(hdc_eval), "hdc_min": min(hdc_eval),
                             "paired_hsf_wins": sum(a > b for a, b in zip(hsf_eval, hdc_eval)),
                             "paired_hdc_wins": sum(b > a for a, b in zip(hsf_eval, hdc_eval))})
    return {"experiment": "hsf_vocabulary_v1", "rho": rho, "select_seeds": list(select_seeds), "eval_seeds": list(eval_seeds), "rows": rows,
            "boundary": "Synthetic correlated bipolar keys; HSF pays H*W usage floats; random value codes; records replayed 3x."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/hsf_vocabulary_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in result["rows"]:
        print(f"V={r['vocab']:<4} B={r['budget']:<6} n={r['count']:<3} HSF {r['hsf_mean']:.2f} (min {r['hsf_min']:.2f}, shape {r['hsf_shape']})  "
              f"HDC {r['hdc_mean']:.2f} (min {r['hdc_min']:.2f})  wins {r['paired_hsf_wins']}/{r['paired_hdc_wins']}")


if __name__ == "__main__":
    main()
