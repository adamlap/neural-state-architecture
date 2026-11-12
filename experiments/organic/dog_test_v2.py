"""Dog test v2: does the trailing field really do the binding, and what happens as the delay grows?

v1 stored the cue-end key and handed it to the learning step after the delay, so the trailing field did not mediate the
association (the v1 write-up overstated this). Here the two mechanisms are separate agents:

* ``trace``  - true trace conditioning: at outcome time the key is the *live* field state (the cue only survives as a decaying
               trace); learn M <- M + eta * err * code(key_now). Prediction at test time uses the cue-end state.
* ``stored`` - eligibility buffer: the cue-end key is remembered and used when the outcome arrives (v1's behaviour).

Field/key variants: ``mid`` (lambda .8), ``slow`` (.95), ``multi`` (concatenation of lambdas .5/.8/.95, each normalised).
Delay sweep: 3, 8, 15, 25 steps. Events are spaced so outcomes do not overlap the next cue.

Part 2b - self-detected events: a linear predictor of the next input from the field state is learned online; cue onsets are the
steps where its error is a large z-score above its running mean. Scored as onset precision/recall against the true onsets, so the
agent no longer needs an oracle for "when did a cue happen".
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from . import dog_test as d1

X = d1.X
LAMBDAS = {"mid": (0.8,), "slow": (0.95,), "multi": (0.5, 0.8, 0.95)}
SPARSE_DIM, SPARSITY, ETA, THETA = 1024, 0.05, 0.5, 0.15


class MultiField:
    def __init__(self, lambdas) -> None:
        self.lambdas = lambdas
        self.z = np.zeros((len(lambdas), X), dtype=np.float32)

    def step(self, x: np.ndarray) -> None:
        for i, lam in enumerate(self.lambdas):
            self.z[i] = lam * self.z[i] + (1 - lam) * x

    def key(self) -> np.ndarray:
        parts = [z / n if (n := np.linalg.norm(z)) > 1e-8 else z for z in self.z]
        return np.concatenate(parts) / np.sqrt(len(parts))


class Sep:
    def __init__(self, seed: int, dim: int) -> None:
        self.w = np.random.default_rng(seed + 9001).standard_normal((SPARSE_DIM, dim)).astype(np.float32)
        self.k = int(round(SPARSE_DIM * SPARSITY))

    def __call__(self, key):
        drive = self.w @ key
        code = np.zeros(SPARSE_DIM, dtype=np.float32)
        code[np.argpartition(drive, -self.k)[-self.k:]] = 1.0
        return code / np.sqrt(self.k)


class Learner:
    def __init__(self, seed: int, variant: str, binding: str) -> None:
        self.lambdas, self.binding = LAMBDAS[variant], binding
        self.sep = Sep(seed, X * len(self.lambdas))
        self.M = np.zeros(SPARSE_DIM, dtype=np.float32)

    def predict(self, key) -> float:
        return float(self.M @ self.sep(key))

    def learn(self, key, outcome: float) -> None:
        code = self.sep(key)
        err = outcome - float(self.M @ code)
        if abs(err) > THETA:
            self.M += ETA * err * code / float(code @ code)


def settle(rng, field: MultiField, steps: int) -> None:
    for _ in range(steps):
        field.step(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32))


def episode_keys(world: d1.World, lambdas, kind: int, delay: int, *, noise=d1.CUE_NOISE, proto=None, scale=1.0):
    """Run one event; return (key at cue end, key at outcome time)."""
    f = MultiField(lambdas)
    settle(world.rng, f, 8)
    base = world.prototypes[kind] if proto is None else proto
    for _ in range(d1.CUE_STEPS):
        f.step(scale * base + noise * world.rng.standard_normal(X).astype(np.float32))
    cue_end = f.key()
    settle(world.rng, f, delay)
    return cue_end, f.key()


def stream_event(world: d1.World, field: MultiField, kind: int, delay: int):
    """One event inside a CONTINUOUS stream: the field is not reset, so traces of earlier events are still present.

    (The first version of this experiment re-created the field for every event, which removed all cross-event contamination and
    inflated the slow-trace results; see docs/ORGANIC_RESEARCH.md.)
    """
    settle(world.rng, field, int(world.rng.integers(*d1.GAP)))
    for _ in range(d1.CUE_STEPS):
        field.step(world.cue_sample(kind, d1.CUE_NOISE))
    cue_end = field.key()
    settle(world.rng, field, delay)
    return cue_end, field.key()


def kind_stream(world: d1.World, n: int):
    for _ in range(n):
        yield int(world.rng.integers(0, d1.N_VALENCED)) if world.rng.random() < 1 / 3 else int(world.rng.integers(d1.N_VALENCED, d1.N_VALENCED + d1.N_DECOY))


def evaluate(agent: Learner, world: d1.World, delay: int) -> dict:
    ok_v, ok_d = [], []
    for kind in range(d1.N_VALENCED + d1.N_DECOY):
        for _ in range(6):
            cue_end, _ = episode_keys(world, agent.lambdas, kind, 0, noise=d1.TEST_NOISE, scale=float(world.rng.uniform(0.7, 1.3)))
            p = agent.predict(cue_end)
            if kind < d1.N_VALENCED:
                ok_v.append(float(np.sign(p) == world.valence[kind] and abs(p) > 0.25))
            else:
                ok_d.append(float(abs(p) < 0.5))
    return {"valenced_accuracy": statistics.fmean(ok_v), "decoy_accuracy": statistics.fmean(ok_d)}


def run_one(binding: str, variant: str, delay: int, seed: int, n_events: int = 400) -> dict:
    world = d1.World.make(seed)
    agent = Learner(seed, variant, binding)
    field = MultiField(agent.lambdas)
    settle(world.rng, field, 20)
    for kind in kind_stream(world, n_events):
        cue_end, at_outcome = stream_event(world, field, kind, delay)
        agent.learn(cue_end if binding == "stored" else at_outcome, float(world.valence[kind]))
    return {"binding": binding, "variant": variant, "delay": delay, "seed": seed, **evaluate(agent, world, delay)}


def detect_onsets(seed: int, n_events: int = 150, warmup: int = 300, z_thresh: float = 4.0) -> dict:
    """Self-detected events from next-input prediction error; no oracle for cue times."""
    world = d1.World.make(seed)
    rng = world.rng
    f = MultiField((0.8,))
    W = np.zeros((X, X), dtype=np.float32)                      # predicts x_{t+1} from the field state
    errs_mean, errs_var, n_seen = 0.0, 1.0, 0
    truth, detected = [], []
    t = 0

    def tick(x, record_truth: bool):
        nonlocal W, errs_mean, errs_var, n_seen, t
        z = f.z[0].copy()
        err_vec = x - W @ z
        err = float(np.linalg.norm(err_vec))
        zs = (err - errs_mean) / (np.sqrt(errs_var) + 1e-6) if n_seen > 50 else 0.0
        if zs > z_thresh and not (detected and t - detected[-1] < d1.CUE_STEPS + 2):
            detected.append(t)
        if zs <= z_thresh:                                      # surprising steps do not update the baseline model
            W += 0.05 * np.outer(err_vec, z) / (float(z @ z) + 1e-6)
            errs_mean += 0.02 * (err - errs_mean)
            errs_var += 0.02 * ((err - errs_mean) ** 2 - errs_var)
            n_seen += 1
        f.step(x)
        t += 1

    for _ in range(warmup):
        tick(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32), False)
    for kind in kind_stream(world, n_events):
        for _ in range(int(rng.integers(*d1.GAP))):
            tick(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32), False)
        truth.append(t)
        for _ in range(d1.CUE_STEPS):
            tick(world.cue_sample(kind, d1.CUE_NOISE), True)
        for _ in range(3):
            tick(d1.BASE_NOISE * rng.standard_normal(X).astype(np.float32), False)
    hit = sum(any(0 <= dt - tt <= 1 for dt in detected) for tt in truth)
    true_pos = sum(any(0 <= dt - tt <= 1 for tt in truth) for dt in detected)
    return {"seed": seed, "events": len(truth), "detected": len(detected), "recall": hit / len(truth),
            "precision": true_pos / len(detected) if detected else 0.0}


def run(*, seeds=tuple(range(8)), delays=(3, 8, 15, 25), variants=("mid", "slow", "multi"), bindings=("stored", "trace")) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    rows = [run_one(b, v, dl, s) for b in bindings for v in variants for dl in delays for s in seeds]
    table = {}
    for b in bindings:
        for v in variants:
            for dl in delays:
                rs = [r for r in rows if (r["binding"], r["variant"], r["delay"]) == (b, v, dl)]
                table[f"{b}|{v}|{dl}"] = {"valenced_accuracy": statistics.fmean(r["valenced_accuracy"] for r in rs),
                                          "decoy_accuracy": statistics.fmean(r["decoy_accuracy"] for r in rs)}
    detect = [detect_onsets(s) for s in seeds]
    return {"experiment": "organic_dog_test_v2", "seeds": list(seeds), "delays": list(delays), "table": table,
            "onset_detection": {"recall": statistics.fmean(r["recall"] for r in detect), "precision": statistics.fmean(r["precision"] for r in detect),
                                "runs": detect}, "runs": rows,
            "boundary": "Toy world; non-overlapping events; oracle cue-end time for the learners (detection is scored separately)."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_dog_test_v2.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("valenced accuracy (decoy accuracy) by delay")
    print(f"{'binding|key':<14}" + "".join(f"{dl:>13}" for dl in result["delays"]))
    for b in ("stored", "trace"):
        for v in ("mid", "slow", "multi"):
            print(f"{b + '|' + v:<14}" + "".join(
                f"{result['table'][f'{b}|{v}|{dl}']['valenced_accuracy']:>8.2f}({result['table'][f'{b}|{v}|{dl}']['decoy_accuracy']:.2f})" for dl in result["delays"]))
    print("self-detected onsets:", {k: round(v, 2) for k, v in result["onset_detection"].items() if k != "runs"})


if __name__ == "__main__":
    main()
