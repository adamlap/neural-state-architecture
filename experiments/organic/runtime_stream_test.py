"""Continuous-stream test through the NSA runtime: learning, uncertainty, and a spoofed sensor.

The dog world (continuous multi-channel input, cue -> delayed outcome, mostly decoys) is driven through ``NSARuntime.observe_signal`` with a
``StreamLearner`` (trailing field -> sparse code -> one fixed delta-rule vector, surprise-gated writes). Three measurements:

1. Learning through the runtime API (valenced-cue accuracy and decoy accuracy on held-out noisy variants) for a fast trace at delay 3 and a slow trace
   at delay 15. Same algorithm as the standalone dog test, so the numbers should agree; this checks that the runtime integration does not change it.
2. Soft-state uncertainty: the runtime's ``state.soft.uncertainty`` (an EMA of absolute prediction error) over training, and its rank correlation with
   the learner's true error rate on held-out cues evaluated at checkpoints. Uncertainty is useful only if it tracks being wrong.
3. Spoofed sensor: with probability 0.5 an event's outcome is sent twice, once truthfully on the trusted channel and once with the sign FLIPPED on a
   second channel ``spoofed``. With ``untrusted_sources=("spoofed",)`` the learner ignores the spoofed outcomes (integrity gate); with the default
   (all channels trusted) it learns from both. Boundary: toy world, hand-set constants, the spoofed channel is labelled untrusted by the ingress path
   (as with memory writes, the label is the trust boundary).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from nsa import NSA, RuntimeConfig
from nsa.organic import StreamLearner

from . import dog_test as d1

X = d1.X


class _Echo:
    model = "echo"

    def generate(self, prompt, **kw):
        return ""


def make_agent(seed: int, lam: float, untrusted: tuple[str, ...] = ()) -> NSA:
    return NSA(_Echo(), config=RuntimeConfig(untrusted_sources=untrusted), stream_learner=StreamLearner(X, lam=lam, seed=seed))


def noise(rng, scale=d1.BASE_NOISE):
    return scale * rng.standard_normal(X).astype(np.float32)


def run_event(agent: NSA, world: d1.World, kind: int, delay: int, *, spoof: bool = False) -> None:
    rng = world.rng
    for _ in range(int(rng.integers(*d1.GAP))):
        agent.observe_signal(noise(rng))
    for _ in range(d1.CUE_STEPS):
        agent.observe_signal(world.cue_sample(kind, d1.CUE_NOISE))
    for _ in range(delay - 1):
        agent.observe_signal(noise(rng))
    agent.observe_signal(noise(rng), outcome=float(world.valence[kind]))                         # trusted channel
    if spoof:
        agent.observe_signal(noise(rng), outcome=-float(world.valence[kind]), source="spoofed")   # sign-flipped, untrusted channel


def evaluate(agent: NSA, world: d1.World) -> dict:
    ok_v, ok_d, errs = [], [], []
    for kind in range(d1.N_VALENCED + d1.N_DECOY):
        for _ in range(6):
            agent.stream_learner.field.z[:] = 0.0
            for _ in range(8):
                agent.observe_signal(noise(world.rng))
            p = 0.0
            scale = float(world.rng.uniform(0.7, 1.3))
            for _ in range(d1.CUE_STEPS):
                p = agent.observe_signal(scale * world.prototypes[kind] + d1.TEST_NOISE * world.rng.standard_normal(X).astype(np.float32))
            errs.append(min(1.0, abs(float(world.valence[kind]) - p)))
            (ok_v if kind < d1.N_VALENCED else ok_d).append(
                float(np.sign(p) == world.valence[kind] and abs(p) > 0.25) if kind < d1.N_VALENCED else float(abs(p) < 0.5))
    return {"valenced_accuracy": statistics.fmean(ok_v), "decoy_accuracy": statistics.fmean(ok_d), "mean_abs_error": statistics.fmean(errs)}


def rank_corr(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else 0.0


def learning(seed: int, lam: float, delay: int, n_events: int = 300, checkpoints=(25, 50, 100, 200, 300)) -> dict:
    world = d1.World.make(seed)
    agent = make_agent(seed, lam)
    curve, unc, err = [], [], []
    done = 0
    for cp in checkpoints:
        for kind in _kinds(world, cp - done):
            run_event(agent, world, kind, delay)
        done = cp
        u = agent.state.soft.uncertainty
        ev = evaluate(agent, world)
        curve.append({"events": cp, **ev})
        unc.append(u)
        err.append(ev["mean_abs_error"])
    return {"lam": lam, "delay": delay, "seed": seed, "curve": curve, "uncertainty_vs_error_rank_corr": rank_corr(unc, err),
            "final": curve[-1]}


def _kinds(world, n):
    for _ in range(n):
        yield int(world.rng.integers(0, d1.N_VALENCED)) if world.rng.random() < 1 / 3 else int(world.rng.integers(d1.N_VALENCED, d1.N_VALENCED + d1.N_DECOY))


def spoof_run(seed: int, gated: bool, lam: float = 0.8, delay: int = 3, n_events: int = 300) -> dict:
    world = d1.World.make(seed)
    agent = make_agent(seed, lam, untrusted=("spoofed",) if gated else ())
    for kind in _kinds(world, n_events):
        run_event(agent, world, kind, delay, spoof=bool(world.rng.random() < 0.5))
    return {"gated": gated, "seed": seed, **evaluate(agent, world), "learning_blocked": agent.signal_stats["learning_blocked"]}


def run(*, seeds=tuple(range(8))) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    setups = ((0.8, 3), (0.95, 15))
    learn = [learning(s, lam, dl) for lam, dl in setups for s in seeds]
    agg = {}
    for lam, dl in setups:
        rs = [r for r in learn if (r["lam"], r["delay"]) == (lam, dl)]
        agg[f"lam={lam},delay={dl}"] = {
            "valenced_accuracy": statistics.fmean(r["final"]["valenced_accuracy"] for r in rs),
            "decoy_accuracy": statistics.fmean(r["final"]["decoy_accuracy"] for r in rs),
            "uncertainty_vs_error_rank_corr": statistics.fmean(r["uncertainty_vs_error_rank_corr"] for r in rs),
            "curve": [statistics.fmean(r["curve"][i]["valenced_accuracy"] for r in rs) for i in range(len(rs[0]["curve"]))]}
    spoof = [spoof_run(s, g) for g in (True, False) for s in seeds]
    spoof_agg = {("gated" if g else "ungated"): {
        "valenced_accuracy": statistics.fmean(r["valenced_accuracy"] for r in spoof if r["gated"] == g),
        "decoy_accuracy": statistics.fmean(r["decoy_accuracy"] for r in spoof if r["gated"] == g),
        "worst_seed_valenced": min(r["valenced_accuracy"] for r in spoof if r["gated"] == g),
        "learning_blocked_mean": statistics.fmean(r["learning_blocked"] for r in spoof if r["gated"] == g)} for g in (True, False)}
    return {"experiment": "organic_runtime_stream_v1", "seeds": list(seeds), "learning": agg, "spoofed_sensor": spoof_agg,
            "runs": {"learning": learn, "spoof": spoof},
            "boundary": "Toy world; the spoofed channel is labelled untrusted at ingress; evaluation uses the learner's prediction at cue end."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_runtime_stream_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for k, v in result["learning"].items():
        print(f"{k:<22} valenced {v['valenced_accuracy']:.2f} decoy {v['decoy_accuracy']:.2f} uncertainty~error rank corr {v['uncertainty_vs_error_rank_corr']:.2f}  curve "
              + " ".join(f"{x:.2f}" for x in v["curve"]))
    for k, v in result["spoofed_sensor"].items():
        print(f"spoofed sensor, {k:<8} valenced {v['valenced_accuracy']:.2f} (worst seed {v['worst_seed_valenced']:.2f}) decoy {v['decoy_accuracy']:.2f} blocked/run {v['learning_blocked_mean']:.0f}")


if __name__ == "__main__":
    main()
