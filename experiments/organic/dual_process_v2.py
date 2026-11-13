"""Dual-process v2: a change detector (CUSUM on queried prediction error) for the uncertainty-gated deliberation policy.

v1 (``dual_process_test``) found that gating queries on familiarity saves queries but does not beat matched random/periodic schedules after a world shift, because
a shifted cue still *looks* familiar. Here the agent watches the prediction error of the queries it does make: err = |outcome - habit prediction|. A one-sided CUSUM
``S = max(0, S + err - mu - slack)`` (mu: slow running mean of err, so ordinary noise is absorbed) alarms when S > ``h``; an alarm opens a window of forced queries
(``alarm_window`` events) and resets S. Compared against ``gated`` (v1), matched-budget ``random`` and ``periodic``.

Protocol: ``slack``/``h``/``alarm_window`` are chosen on development seeds 0-9 only (small grid); the chosen setting is then scored ONCE on unseen seeds 200-209.
Boundary: toy world, perfect oracle, one kind of shift (half of the valenced cues flip once).
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from nsa import NSA
from nsa.organic import StreamLearner

from . import dog_test as d1
from . import dual_process_test as v1
from .dual_process_test import N_EVENTS, SHIFT, X, _Null, _correct, _kinds, summarize_policy

MU_RATE = 0.02


def run_cusum(seed: int, *, slack: float, h: float, alarm_window: int, shift: bool = True, trace: bool = False) -> dict:
    world = d1.World.make(seed)
    agent = NSA(_Null(), stream_learner=StreamLearner(X, lam=0.8, seed=seed))
    rng = world.rng
    plan = list(_kinds(world, N_EVENTS))
    ok, valenced, queries, post, alarm_left, S, mu, alarms = [], [], 0, 0, 0, 0.0, 0.0, []
    for i, kind in enumerate(plan):
        if shift and i == SHIFT:
            world.valence[np.arange(d1.N_VALENCED)[: d1.N_VALENCED // 2 + 1:2]] *= -1.0
        for _ in range(int(rng.integers(*d1.GAP))):
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
        pred = 0.0
        for _ in range(d1.CUE_STEPS):
            pred = agent.observe_signal(world.cue_sample(kind, d1.CUE_NOISE))
        ok.append(_correct(world, kind, pred))
        valenced.append(kind < d1.N_VALENCED)
        familiarity = agent.stream_learner.familiarity()
        ask = familiarity < v1.FAMILIARITY_TAU or alarm_left > 0 or rng.random() < v1.AUDIT
        alarm_left = max(0, alarm_left - 1)
        for _ in range(v1.DELAY - 1):
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
        if ask:
            queries += 1
            post += 1 if i >= SHIFT else 0
            before = agent.stream_learner.predict()
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32), outcome=float(world.valence[kind]))
            if kind < d1.N_VALENCED:                                                    # only valenced cues carry a meaningful outcome error
                err = abs(float(world.valence[kind]) - before)
                S = max(0.0, S + err - mu - slack)
                mu += MU_RATE * (err - mu)
                if S > h:
                    alarm_left, S = alarm_window, 0.0
                    alarms.append(i)
        else:
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
    return {"policy": "cusum", "seed": seed, "queries": queries, "queries_after_shift": post, "ok": ok, "valenced": valenced, "alarms": alarms}


def false_alarms(runs) -> tuple[float, float | None]:
    """(mean alarms before the shift per run, median delay of the first alarm after the shift)."""
    pre = statistics.fmean(sum(a < SHIFT for a in r["alarms"]) for r in runs)
    first = [min((a for a in r["alarms"] if a >= SHIFT), default=None) for r in runs]
    delays = [f - SHIFT for f in first if f is not None]
    return pre, (statistics.median(delays) if delays else None)


def evaluate(seeds, params) -> dict:
    cus = [run_cusum(s, **params) for s in seeds]
    b = statistics.fmean(r["queries"] for r in cus)
    out = {"cusum": summarize_policy(cus)}
    out["random"] = summarize_policy([v1.run_policy("random", s, p=r["queries"] / N_EVENTS) for s, r in zip(seeds, cus)])
    out["periodic"] = summarize_policy([v1.run_policy("periodic", s, k=max(1, round(N_EVENTS / max(1, r["queries"])))) for s, r in zip(seeds, cus)])
    out["gated"] = summarize_policy([v1.run_policy("gated", s) for s in seeds])
    out["alarms_before_shift"], out["first_alarm_delay_median"] = false_alarms(cus)
    out["cusum_runs"] = cus
    out["budget"] = b
    return out


def objective(res) -> float:
    """Dev selection score: post-shift accuracy of cusum minus the matched-budget random baseline (what v1 failed to deliver), penalising false alarms."""
    c, r = res["cusum"], res["random"]
    return (c["acc_right_after_shift"] - r["acc_right_after_shift"]) + 0.5 * (c["acc_end"] - r["acc_end"]) - 0.01 * res["alarms_before_shift"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev-seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--final-seeds", nargs="+", type=int, default=list(range(200, 210)))
    p.add_argument("--out", default="results/organic_dual_process_v2.json")
    p.add_argument("--final", action="store_true", help="score the chosen setting on the unseen seeds (run once)")
    args = p.parse_args()
    grid = [dict(slack=s, h=h, alarm_window=w) for s, h, w in itertools.product((0.1, 0.3), (1.0, 2.0, 4.0), (10, 25))]
    dev = []
    for params in grid:
        res = evaluate(args.dev_seeds, params)
        dev.append({"params": params, "objective": objective(res), "budget": res["budget"], "summary": {k: res[k] for k in ("cusum", "random", "periodic", "gated")},
                    "alarms_before_shift": res["alarms_before_shift"], "first_alarm_delay_median": res["first_alarm_delay_median"]})
        c, r = res["cusum"], res["random"]
        print(f"{params} obj {objective(res):+.3f} budget {res['budget']:.0f} cusum just-after {c['acc_right_after_shift']:.2f} end {c['acc_end']:.2f} | "
              f"random {r['acc_right_after_shift']:.2f} {r['acc_end']:.2f} | false alarms {res['alarms_before_shift']:.1f} delay {res['first_alarm_delay_median']}", flush=True)
    best = max(dev, key=lambda d: d["objective"])
    result = {"experiment": "organic_dual_process_v2", "dev_seeds": args.dev_seeds, "dev": dev, "chosen": best["params"]}
    if args.final:
        res = evaluate(args.final_seeds, best["params"])
        result["final_seeds"] = args.final_seeds
        result["final"] = {k: res[k] for k in ("cusum", "random", "periodic", "gated", "budget", "alarms_before_shift", "first_alarm_delay_median")}
        print("FINAL (unseen seeds)", best["params"])
        for pol in ("cusum", "gated", "random", "periodic"):
            v = res[pol]
            print(f"{pol:<9} queries {v['queries']:.0f} after {v['queries_after_shift']:.0f} pre {v['acc_before_shift']:.2f} just-after {v['acc_right_after_shift']:.2f} end {v['acc_end']:.2f} recovery {v['recovery_events_median']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
