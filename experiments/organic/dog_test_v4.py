"""Dog test v4: selecting timescales by signal-to-noise in a mixed-delay world (delays 3, 15, 40, 80, 150).

v3 found that combining timescales by concatenation or online expert weighting did not beat the best single trace, because at outcome time the fast
traces hold only noise. Hypothesis tested here: let a trace take part only when its norm stands clearly above its own idle noise floor, i.e. when it still holds a
recent event. No learning is involved in the selection.

Learners (same trailing field, sparse codes, delta-rule vector and surprise gate; same events and seeds):
* ``single_<lam>``  - one trace;
* ``joint``         - one sparse code per trace, concatenated, ONE delta-rule vector over all (v3's best);
* ``snr_gated``     - as ``joint`` but each trace's code is used only if ``||z_i|| > ratio * floor_i``; ``floor_i`` is an exponential average of ``||z_i||`` that is
                      only updated while the trace is near its floor (robust to events), so it tracks idle noise.
At test time a fresh field is run over a cue (8 settle steps + 4 cue steps) and the learner predicts at cue end.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from . import dog_test as d1
from . import dog_test_v2 as v2

X = d1.X
DELAYS = (3, 15, 40, 80, 150)
LAMS = (0.8, 0.95, 0.99, 0.995)
SPARSE, SPARSITY, ETA, THETA, RATIO = 1024, 0.05, 0.5, 0.15, 1.5


class Learner:
    def __init__(self, kind: str, seed: int, lams=LAMS) -> None:
        self.kind = kind
        self.lams = (float(kind.split("_")[1]),) if kind.startswith("single_") else tuple(lams)
        self.seps = [v2.Sep(seed + i, X) for i in range(len(self.lams))]
        self.M = np.zeros(SPARSE * len(self.lams), dtype=np.float32)
        self.field = v2.MultiField(self.lams)
        self.floor = np.full(len(self.lams), 1e-3)
        self.k = self.seps[0].k

    def step(self, x) -> None:
        self.field.step(x)
        norms = np.linalg.norm(self.field.z, axis=1)
        near = norms < 1.25 * self.floor
        self.floor = np.where(near, self.floor + 0.02 * (norms - self.floor), np.where(self.floor < 1e-2, norms, self.floor))

    def _code(self) -> np.ndarray:
        parts = []
        norms = np.linalg.norm(self.field.z, axis=1)
        for i, (sep, z) in enumerate(zip(self.seps, self.field.z)):
            use = self.kind != "snr_gated" or norms[i] > RATIO * self.floor[i]
            parts.append(sep(z / max(np.linalg.norm(z), 1e-8)) if use else np.zeros(SPARSE, dtype=np.float32))
        return np.concatenate(parts) / np.sqrt(len(self.lams))

    def predict(self) -> float:
        return float(self.M @ self._code())

    def learn(self, outcome: float) -> None:
        code = self._code()
        n = float(code @ code)
        if n <= 0:
            return
        err = outcome - float(self.M @ code)
        if abs(err) > THETA:
            self.M += ETA * err * code / n


def noise(rng, s=d1.BASE_NOISE):
    return s * rng.standard_normal(X).astype(np.float32)


def train(learner: Learner, world, delays: dict, n_events: int) -> None:
    rng = world.rng
    for _ in range(40):
        learner.step(noise(rng))
    for _ in range(n_events):
        kind = int(rng.integers(0, d1.N_VALENCED)) if rng.random() < 1 / 3 else int(rng.integers(d1.N_VALENCED, d1.N_VALENCED + d1.N_DECOY))
        for _ in range(int(rng.integers(*d1.GAP))):
            learner.step(noise(rng))
        for _ in range(d1.CUE_STEPS):
            learner.step(world.cue_sample(kind, d1.CUE_NOISE))
        for _ in range(delays.get(kind, 3)):
            learner.step(noise(rng))
        learner.learn(float(world.valence[kind]))


def evaluate(learner: Learner, world, delays: dict) -> dict:
    keep_z, keep_floor = learner.field.z.copy(), learner.floor.copy()
    ok_v, ok_d, per = [], [], {d: [] for d in DELAYS}
    for kind in range(d1.N_VALENCED + d1.N_DECOY):
        for _ in range(6):
            learner.field.z[:] = 0.0
            for _ in range(8):
                learner.step(noise(world.rng))
            scale = float(world.rng.uniform(0.7, 1.3))
            for _ in range(d1.CUE_STEPS):
                learner.step(scale * world.prototypes[kind] + d1.TEST_NOISE * world.rng.standard_normal(X).astype(np.float32))
            p = learner.predict()
            if kind < d1.N_VALENCED:
                ok = float(np.sign(p) == world.valence[kind] and abs(p) > 0.25)
                ok_v.append(ok)
                per[delays[kind]].append(ok)
            else:
                ok_d.append(float(abs(p) < 0.5))
    learner.field.z[:], learner.floor[:] = keep_z, keep_floor
    return {"valenced": statistics.fmean(ok_v), "decoy": statistics.fmean(ok_d), "by_delay": {str(k): statistics.fmean(v) for k, v in per.items() if v}}


def run_one(kind: str, seed: int, n_events: int = 600) -> dict:
    world = d1.World.make(seed)
    delays = {k: DELAYS[k % len(DELAYS)] for k in range(d1.N_VALENCED)}
    learner = Learner(kind, seed)
    train(learner, world, delays, n_events)
    return {"learner": kind, "seed": seed, **evaluate(learner, world, delays)}


LEARNERS = ("single_0.8", "single_0.95", "single_0.99", "single_0.995", "joint", "snr_gated")


def run(*, seeds=tuple(range(8)), learners=LEARNERS) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    rows = [run_one(k, s) for k in learners for s in seeds]
    table = {k: {"valenced": statistics.fmean(r["valenced"] for r in rows if r["learner"] == k),
                 "decoy": statistics.fmean(r["decoy"] for r in rows if r["learner"] == k),
                 "by_delay": {str(d): statistics.fmean(r["by_delay"][str(d)] for r in rows if r["learner"] == k and str(d) in r["by_delay"]) for d in DELAYS}}
             for k in learners}
    return {"experiment": "organic_dog_test_v4", "seeds": list(seeds), "delays": list(DELAYS), "lams": list(LAMS), "ratio": RATIO, "table": table, "runs": rows,
            "boundary": "Toy world; per-cue-type delays fixed; oracle outcome time; hand-set noise-floor rule."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_dog_test_v4.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'learner':<14}{'overall':>8}{'decoy':>7}" + "".join(f"{'d=' + str(d):>7}" for d in DELAYS))
    for k, v in result["table"].items():
        print(f"{k:<14}{v['valenced']:>8.2f}{v['decoy']:>7.2f}" + "".join(f"{v['by_delay'][str(d)]:>7.2f}" for d in DELAYS))


if __name__ == "__main__":
    main()
