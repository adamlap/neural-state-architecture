"""Dual-process test: does uncertainty decide where costly deliberation is spent?

The dog world is driven through ``NSARuntime.observe_signal``. Feedback (the outcome of a cue) only arrives when the agent *queries* (a costly deliberation: an
LLM call, an expert, a human). Halfway through, half of the rewarding/aversive cues flip meaning, so habit learned earlier becomes wrong and only fresh feedback fixes it.
At the end of every cue the agent has a habit prediction (free) and decides whether to query. Policies:

* ``always``   - query every event (the cost ceiling and the accuracy ceiling);
* ``never``    - never query (no learning);
* ``random``   - query with probability p; ``periodic`` - every k-th event; both matched to the gated policy's query count on the same seed;
* ``gated``    - query when the input is unfamiliar (``StreamLearner.familiarity`` below a threshold), plus a small random audit rate, plus a burst of forced queries after
                 a query reveals a large prediction error (surprise). This is the only policy that uses the agent's own uncertainty.

Scores are online and are computed on the VALENCED cues only (decoys are two thirds of events and a policy that never learns already gets them right): the habit prediction at cue end, judged against the truth at that moment, before the event's own feedback. Reported: accuracy in the window before the shift,
accuracy right after it and at the end, queries used overall and after the shift, and recovery time (events after the shift until a 30-event window of valenced cues reaches 0.9).
Boundary: toy world; the oracle is perfect; constants hand-set; not tuned per policy.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from nsa import NSA
from nsa.organic import StreamLearner

from . import dog_test as d1

X, DELAY, N_EVENTS, SHIFT = d1.X, 3, 600, 300
FAMILIARITY_TAU, AUDIT, BURST, SURPRISE = 0.5, 0.03, 6, 0.5
POLICIES = ("always", "never", "gated", "gated_state", "random", "periodic")
AUDIT_GAIN, UNCERTAINTY_FLOOR, AUDIT_MAX = 0.8, 0.1, 0.6


class _Null:
    model = "null"

    def generate(self, prompt, **kw):
        return ""


def _kinds(world, n):
    for _ in range(n):
        yield int(world.rng.integers(0, d1.N_VALENCED)) if world.rng.random() < 1 / 3 else int(world.rng.integers(d1.N_VALENCED, d1.N_VALENCED + d1.N_DECOY))


def _correct(world, kind: int, p: float) -> bool:
    return bool(np.sign(p) == world.valence[kind] and abs(p) > 0.25) if kind < d1.N_VALENCED else bool(abs(p) < 0.5)


def run_policy(policy: str, seed: int, *, p: float = 0.2, k: int = 5) -> dict:
    world = d1.World.make(seed)
    agent = NSA(_Null(), stream_learner=StreamLearner(X, lam=0.8, seed=seed))
    rng = world.rng
    plan = list(_kinds(world, N_EVENTS))
    flipped = False
    ok, valenced, queries, burst, post_queries = [], [], 0, 0, 0
    for i, kind in enumerate(plan):
        if i == SHIFT and not flipped:
            flip = np.arange(d1.N_VALENCED)[: d1.N_VALENCED // 2 + 1:2]                  # flip half of the valenced cues (indices 0, 2, 4)
            world.valence[flip] *= -1.0
            flipped = True
        for _ in range(int(rng.integers(*d1.GAP))):
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
        pred = 0.0
        for _ in range(d1.CUE_STEPS):
            pred = agent.observe_signal(world.cue_sample(kind, d1.CUE_NOISE))
        ok.append(_correct(world, kind, pred))
        valenced.append(kind < d1.N_VALENCED)
        familiarity = agent.stream_learner.familiarity()
        if policy == "always":
            ask = True
        elif policy == "never":
            ask = False
        elif policy == "gated":
            ask = familiarity < FAMILIARITY_TAU or burst > 0 or rng.random() < AUDIT
            burst = max(0, burst - 1)
        elif policy == "gated_state":
            # as "gated", but the audit rate follows the RUNTIME's soft-state uncertainty (EMA of recent |prediction error| on queried outcomes): a shift makes
            # errors large, uncertainty rises, and the agent starts re-checking cues it thought it knew
            u = float(agent.state.soft.uncertainty)
            audit = min(AUDIT_MAX, AUDIT + AUDIT_GAIN * max(0.0, u - UNCERTAINTY_FLOOR))
            ask = familiarity < FAMILIARITY_TAU or burst > 0 or rng.random() < audit
            burst = max(0, burst - 1)
        elif policy == "random":
            ask = rng.random() < p
        else:
            ask = i % k == 0
        for _ in range(DELAY - 1):
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
        if ask:
            queries += 1
            post_queries += 1 if i >= SHIFT else 0
            before = agent.stream_learner.predict()
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32), outcome=float(world.valence[kind]))
            if policy in ("gated", "gated_state") and abs(float(world.valence[kind]) - before) > SURPRISE:
                burst = BURST
        else:
            agent.observe_signal(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))
    return {"policy": policy, "seed": seed, "queries": queries, "queries_after_shift": post_queries, "ok": ok, "valenced": valenced}


def window(ok, a, b, mask=None) -> float:
    """Mean correctness over events a..b; with ``mask`` only over the events where it is True (the valenced cues: decoys are 2/3 of events and trivially 'correct')."""
    vals = [o for i, o in enumerate(ok[a:b], a) if mask is None or mask[i]]
    return statistics.fmean(vals) if vals else float("nan")


def recovery(ok, mask=None) -> int | None:
    for i in range(SHIFT, len(ok) - 30):
        if window(ok, i, i + 30, mask) >= 0.9:
            return i - SHIFT
    return None


def summarize_policy(runs) -> dict:
    rec = [recovery(r["ok"], r["valenced"]) for r in runs]
    return {"queries": statistics.fmean(r["queries"] for r in runs), "queries_after_shift": statistics.fmean(r["queries_after_shift"] for r in runs),
            "acc_before_shift": statistics.fmean(window(r["ok"], 200, SHIFT, r["valenced"]) for r in runs),
            "acc_right_after_shift": statistics.fmean(window(r["ok"], SHIFT, SHIFT + 90, r["valenced"]) for r in runs),
            "acc_end": statistics.fmean(window(r["ok"], N_EVENTS - 150, N_EVENTS, r["valenced"]) for r in runs),
            "decoy_acc_end": statistics.fmean(window(r["ok"], N_EVENTS - 150, N_EVENTS, [not v for v in r["valenced"]]) for r in runs),
            "recovery_events_median": statistics.median([x for x in rec if x is not None]) if any(x is not None for x in rec) else None,
            "recovered_share": sum(x is not None for x in rec) / len(rec)}


def run(*, seeds=tuple(range(10))) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    out, runs = {}, {}
    gated = [run_policy("gated", s) for s in seeds]
    runs["gated"] = gated
    for pol in ("always", "never"):
        runs[pol] = [run_policy(pol, s) for s in seeds]
    # matched baselines: same query budget as the gated policy on the same seed
    runs["gated_state"] = [run_policy("gated_state", s) for s in seeds]
    # matched baselines: the same query budget as the corresponding gated policy on the same seed
    runs["random"] = [run_policy("random", s, p=g["queries"] / N_EVENTS) for s, g in zip(seeds, runs["gated_state"])]
    runs["periodic"] = [run_policy("periodic", s, k=max(1, round(N_EVENTS / max(1, g["queries"])))) for s, g in zip(seeds, runs["gated_state"])]
    runs["random_vs_gated"] = [run_policy("random", s, p=g["queries"] / N_EVENTS) for s, g in zip(seeds, gated)]
    for pol in POLICIES:
        out[pol] = summarize_policy(runs[pol])
    out["random_vs_gated"] = summarize_policy(runs["random_vs_gated"])
    return {"experiment": "organic_dual_process_v1", "seeds": list(seeds), "events": N_EVENTS, "shift_at": SHIFT, "summary": out,
            "constants": {"familiarity_tau": FAMILIARITY_TAU, "audit": AUDIT, "burst": BURST, "surprise": SURPRISE},
            "boundary": "Toy world, perfect oracle, hand-set constants; query counts matched to the gated policy for random and periodic."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_dual_process_v1.json")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = p.parse_args()
    result = run(seeds=tuple(args.seeds))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'policy':<10}{'queries':>8}{'after shift':>12}{'acc pre':>9}{'acc just after':>16}{'acc end':>9}{'recovery':>10}{'recovered':>11}   (valenced cues)")
    for pol, v in result["summary"].items():
        print(f"{pol:<10}{v['queries']:>8.0f}{v['queries_after_shift']:>12.0f}{v['acc_before_shift']:>9.2f}{v['acc_right_after_shift']:>16.2f}{v['acc_end']:>9.2f}"
              f"{str(v['recovery_events_median']):>10}{v['recovered_share']:>11.0%}")


if __name__ == "__main__":
    main()
