"""Stage 4d: does the Stage 4c win survive tuning the baseline?

Sweeps, for BOTH methods, the separator sparsity and (for HSF) the field shape at a fixed 4,096-float field,
over 10 seeds, at within-family cosine 0.9 and 0.5. HDC gets its own best sparsity; HSF gets its best (sparsity, shape).
A fair verdict compares each method's best configuration, chosen on seeds that are NOT the evaluation seeds.

Selection seeds and evaluation seeds are disjoint (``SELECT_SEEDS`` vs ``EVAL_SEEDS``), so the reported numbers are
not inflated by choosing the best cell on the same data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import consolidation_benchmark as cb

SELECT_SEEDS = (1, 2, 3, 4, 5)
EVAL_SEEDS = (11, 23, 47, 59, 71, 83, 97, 101, 113, 127)
SPARSITIES = (0.02, 0.05, 0.10, 0.20)
SHAPES = ((16, 16, 16), (16, 32, 8), (32, 32, 4))   # 4,096 field floats each; channels >= vocabulary size is enforced below
HSF, HDC = "hsf_sep_homeo_replay", "hdc_sep_replay"


def _accuracies(method, seeds, n, rho, shape, sparsity) -> list[float]:
    old = cb.SPARSITY
    cb.SPARSITY = sparsity
    try:
        return [cb.run_one(method, s, n, rho, *shape)["overall_accuracy"] for s in seeds]
    finally:
        cb.SPARSITY = old


def _score(method, seeds, n, rho, shape, sparsity) -> float:
    return statistics.fmean(_accuracies(method, seeds, n, rho, shape, sparsity))


def run(*, counts=(64, 128), similarities=(0.9, 0.5), select_seeds=SELECT_SEEDS, eval_seeds=EVAL_SEEDS,
        sparsities=SPARSITIES, shapes=SHAPES) -> dict:
    if set(select_seeds) & set(eval_seeds):
        raise ValueError("selection and evaluation seeds must be disjoint")
    shapes = [s for s in shapes if s[2] >= len(cb.VOCABULARY)]          # a field needs C >= vocabulary size
    flat = shapes[0][0] * shapes[0][1] * shapes[0][2]
    rows = []
    for rho in similarities:
        for n in counts:
            hsf_grid = {(sh, sp): _score(HSF, select_seeds, n, rho, sh, sp) for sh in shapes for sp in sparsities}
            hdc_grid = {sp: _score(HDC, select_seeds, n, rho, shapes[0], sp) for sp in sparsities}
            (bsh, bsp), hsf_sel = max(hsf_grid.items(), key=lambda kv: kv[1])
            bdp = max(hdc_grid, key=hdc_grid.get)
            hsf_eval = _accuracies(HSF, eval_seeds, n, rho, bsh, bsp)
            hdc_eval = _accuracies(HDC, eval_seeds, n, rho, shapes[0], bdp)
            wins = sum(a > b for a, b in zip(hsf_eval, hdc_eval))
            ties = sum(a == b for a, b in zip(hsf_eval, hdc_eval))
            rows.append({"rho": rho, "count": n, "hsf_best_shape": list(bsh), "hsf_best_sparsity": bsp,
                         "hdc_best_sparsity": bdp, "hsf_select": hsf_sel, "hdc_select": hdc_grid[bdp],
                         "hsf_eval_mean": statistics.fmean(hsf_eval), "hdc_eval_mean": statistics.fmean(hdc_eval),
                         "hsf_eval_min": min(hsf_eval), "hdc_eval_min": min(hdc_eval),
                         "paired_wins_hsf": wins, "paired_ties": ties, "paired_wins_hdc": len(eval_seeds) - wins - ties})
    return {"experiment": "hsf_replication_v1", "select_seeds": list(select_seeds), "eval_seeds": list(eval_seeds),
            "sparsities": list(sparsities), "shapes": [list(s) for s in shapes], "field_floats": flat, "rows": rows,
            "boundary": "Synthetic correlated bipolar keys, 8-value vocabulary, all records replayed 3 times; HSF also pays H*W usage floats."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/hsf_replication_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in result["rows"]:
        print(f"rho={r['rho']} n={r['count']}: HSF {r['hsf_eval_mean']:.2f} (min {r['hsf_eval_min']:.2f}, shape {r['hsf_best_shape']}, sp {r['hsf_best_sparsity']})"
              f"  vs HDC {r['hdc_eval_mean']:.2f} (min {r['hdc_eval_min']:.2f}, sp {r['hdc_best_sparsity']})"
              f"  paired HSF/tie/HDC = {r['paired_wins_hsf']}/{r['paired_ties']}/{r['paired_wins_hdc']}")


if __name__ == "__main__":
    main()
