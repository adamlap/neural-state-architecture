"""Dog test v3: (1) how robust is self-detection of events, (2) how should timescales be combined?

Part 1 - detection robustness. ``detect_onsets`` (next-input prediction error z-score, learned online) vs a plain detector (z-score of the
input norm against its running mean), swept over cue amplitude (signal) and baseline noise.

Part 2 - timescale combination in a *mixed-delay world*: cue types have different outcome delays (3, 15, 40, 80), so no single trace time
constant fits all. Trace-bound learners (the key is the live field state at outcome time) compared:
* ``single_<lambda>``   - one trace (.8, .95, .99);
* ``concat_pre``        - v2's failing equal-weight sum of normalised traces, then one sparse code;
* ``joint_codes``       - each timescale gets its own sparse code; the codes are concatenated and ONE delta-rule vector learns over them,
                          so the learner assigns the weights implicitly (a unit that is useful at one delay is used there only);
* ``expert_mix``        - one delta-rule expert per timescale, predictions combined by weights exp(-beta * running error) (online
                          exponentiated-gradient style selection).
Scores: valenced and decoy accuracy at cue end. Everything uses the same events, noise and seeds.
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
DELAYS = (3, 15, 40, 80)
TRACE_SET = (0.8, 0.95, 0.99)


# ---------------------------------------------------------------- part 1: detection robustness
def _stream(seed: int, cue_scale: float, base_noise: float, n_events: int):
    world = d1.World.make(seed)
    rng = world.rng
    xs, onsets = [], []
    for _ in range(300):
        xs.append(base_noise * rng.standard_normal(X).astype(np.float32))
    for kind in v2.kind_stream(world, n_events):
        for _ in range(int(rng.integers(*d1.GAP))):
            xs.append(base_noise * rng.standard_normal(X).astype(np.float32))
        onsets.append(len(xs))
        for _ in range(d1.CUE_STEPS):
            xs.append(cue_scale * world.prototypes[kind] + d1.CUE_NOISE * base_noise / 0.2 * rng.standard_normal(X).astype(np.float32))
        for _ in range(3):
            xs.append(base_noise * rng.standard_normal(X).astype(np.float32))
    return xs, onsets


def _score(detected: list[int], truth: list[int]) -> tuple[float, float]:
    hit = sum(any(0 <= dt - tt <= 1 for dt in detected) for tt in truth)
    tp = sum(any(0 <= dt - tt <= 1 for tt in truth) for dt in detected)
    return hit / len(truth), (tp / len(detected) if detected else 0.0)


def detect_predictive(xs, z_thresh: float) -> list[int]:
    f = v2.MultiField((0.8,))
    W = np.zeros((X, X), dtype=np.float32)
    mean, var, n_seen, detected = 0.0, 1.0, 0, []
    for t, x in enumerate(xs):
        z = f.z[0].copy()
        err_vec = x - W @ z
        err = float(np.linalg.norm(err_vec))
        zs = (err - mean) / (np.sqrt(var) + 1e-6) if n_seen > 50 else 0.0
        if zs > z_thresh and not (detected and t - detected[-1] < d1.CUE_STEPS + 2):
            detected.append(t)
        if zs <= z_thresh:
            W += 0.05 * np.outer(err_vec, z) / (float(z @ z) + 1e-6)
            mean += 0.02 * (err - mean)
            var += 0.02 * ((err - mean) ** 2 - var)
            n_seen += 1
        f.step(x)
    return detected


def detect_norm(xs, z_thresh: float) -> list[int]:
    mean, var, n_seen, detected = 0.0, 1.0, 0, []
    for t, x in enumerate(xs):
        v = float(np.linalg.norm(x))
        zs = (v - mean) / (np.sqrt(var) + 1e-6) if n_seen > 50 else 0.0
        if zs > z_thresh and not (detected and t - detected[-1] < d1.CUE_STEPS + 2):
            detected.append(t)
        if zs <= z_thresh:
            mean += 0.02 * (v - mean)
            var += 0.02 * ((v - mean) ** 2 - var)
            n_seen += 1
    return detected


def detection_sweep(*, seeds=tuple(range(4)), scales=(0.25, 0.5, 1.0, 1.5), noises=(0.2, 0.5, 0.8), n_events=80, z_thresh=4.0) -> dict:
    rows = []
    for scale in scales:
        for noise in noises:
            for name, fn in (("predictive", detect_predictive), ("norm", detect_norm)):
                rs = []
                for s in seeds:
                    xs, truth = _stream(s, scale, noise, n_events)
                    rs.append(_score(fn(xs, z_thresh), truth))
                rows.append({"detector": name, "cue_scale": scale, "base_noise": noise,
                             "recall": statistics.fmean(r[0] for r in rs), "precision": statistics.fmean(r[1] for r in rs)})
    return {"rows": rows, "z_thresh": z_thresh}


# ---------------------------------------------------------------- part 2: timescale combination
class MixedWorld(d1.World):
    @classmethod
    def make(cls, seed: int):
        w = super().make(seed)
        w.delays = {k: DELAYS[k % len(DELAYS)] for k in range(d1.N_VALENCED)}          # type: ignore[attr-defined]
        return w




class Single:
    def __init__(self, seed, lam):
        self.lambdas, self.l = (lam,), v2.Learner(seed, "mid", "trace")
        self.l.lambdas, self.l.sep = (lam,), v2.Sep(seed, X)

    def predict(self, key): return self.l.predict(key)
    def learn(self, key, o): self.l.learn(key, o)


class ConcatPre:
    def __init__(self, seed):
        self.lambdas, self.l = TRACE_SET, v2.Learner(seed, "multi", "trace")
        self.l.lambdas = TRACE_SET

    def predict(self, key): return self.l.predict(key)
    def learn(self, key, o): self.l.learn(key, o)


class JointCodes:
    def __init__(self, seed):
        self.lambdas = TRACE_SET
        self.seps = [v2.Sep(seed + i, X) for i in range(len(TRACE_SET))]
        self.M = np.zeros(v2.SPARSE_DIM * len(TRACE_SET), dtype=np.float32)

    def code(self, key):
        parts = np.split(key * np.sqrt(len(self.lambdas)), len(self.lambdas))
        return np.concatenate([s(p / max(np.linalg.norm(p), 1e-8)) for s, p in zip(self.seps, parts)]) / np.sqrt(len(self.lambdas))

    def predict(self, key): return float(self.M @ self.code(key))

    def learn(self, key, o):
        code = self.code(key)
        err = o - float(self.M @ code)
        if abs(err) > v2.THETA:
            self.M += v2.ETA * err * code / float(code @ code)


class ExpertMix:
    BETA = 6.0

    def __init__(self, seed):
        self.lambdas = TRACE_SET
        self.seps = [v2.Sep(seed + i, X) for i in range(len(TRACE_SET))]
        self.Ms = [np.zeros(v2.SPARSE_DIM, dtype=np.float32) for _ in TRACE_SET]
        self.err = np.ones(len(TRACE_SET))

    def _parts(self, key):
        parts = np.split(key * np.sqrt(len(self.lambdas)), len(self.lambdas))
        return [p / max(np.linalg.norm(p), 1e-8) for p in parts]

    def predict(self, key):
        w = np.exp(-self.BETA * self.err)
        w /= w.sum()
        return float(sum(wi * (M @ s(p)) for wi, M, s, p in zip(w, self.Ms, self.seps, self._parts(key))))

    def learn(self, key, o):
        for i, (M, s, p) in enumerate(zip(self.Ms, self.seps, self._parts(key))):
            code = s(p)
            e = o - float(M @ code)
            self.err[i] += 0.1 * (abs(e) - self.err[i])
            if abs(e) > v2.THETA:
                M += v2.ETA * e * code / float(code @ code)


def make_learner(name: str, seed: int):
    if name.startswith("single_"):
        return Single(seed, float(name.split("_")[1]))
    return {"concat_pre": ConcatPre, "joint_codes": JointCodes, "expert_mix": ExpertMix}[name](seed)


LEARNERS = ("single_0.8", "single_0.95", "single_0.99", "concat_pre", "joint_codes", "expert_mix")


def evaluate(agent, world) -> dict:
    ok_v, ok_d, per_delay = [], [], {dl: [] for dl in DELAYS}
    for kind in range(d1.N_VALENCED + d1.N_DECOY):
        for _ in range(6):
            cue_end, _ = v2.episode_keys(world, agent.lambdas, kind, 0, noise=d1.TEST_NOISE, scale=float(world.rng.uniform(0.7, 1.3)))
            p = agent.predict(cue_end)
            if kind < d1.N_VALENCED:
                ok = float(np.sign(p) == world.valence[kind] and abs(p) > 0.25)
                ok_v.append(ok)
                per_delay[world.delays[kind]].append(ok)
            else:
                ok_d.append(float(abs(p) < 0.5))
    return {"valenced": statistics.fmean(ok_v), "decoy": statistics.fmean(ok_d),
            "by_delay": {str(k): statistics.fmean(v) for k, v in per_delay.items() if v}}


def run_one(name: str, seed: int, n_events: int = 600) -> dict:
    world = MixedWorld.make(seed)
    agent = make_learner(name, seed)
    field = v2.MultiField(agent.lambdas)
    v2.settle(world.rng, field, 20)
    for kind in v2.kind_stream(world, n_events):
        _, at_outcome = v2.stream_event(world, field, kind, world.delays.get(kind, 3))
        agent.learn(at_outcome, float(world.valence[kind]))
    return {"learner": name, "seed": seed, **evaluate(agent, world)}


def run(*, seeds=tuple(range(6)), learners=LEARNERS, detection=True) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    rows = [run_one(n, s) for n in learners for s in seeds]
    table = {n: {"valenced": statistics.fmean(r["valenced"] for r in rows if r["learner"] == n),
                 "decoy": statistics.fmean(r["decoy"] for r in rows if r["learner"] == n),
                 "by_delay": {dl: statistics.fmean(r["by_delay"][dl] for r in rows if r["learner"] == n) for dl in map(str, DELAYS)}}
             for n in learners}
    return {"experiment": "organic_dog_test_v3", "seeds": list(seeds), "delays": list(DELAYS), "table": table,
            "detection": detection_sweep(seeds=seeds[:4]) if detection else None, "runs": rows,
            "boundary": "Toy world; per-cue-type delays fixed; oracle cue-end time for the learners."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_dog_test_v3.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("mixed-delay world: valenced accuracy (decoy) | accuracy by cue delay")
    print(f"{'learner':<14}{'overall':>8}{'decoy':>7}" + "".join(f"{'d=' + str(d):>7}" for d in DELAYS))
    for n, v in result["table"].items():
        print(f"{n:<14}{v['valenced']:>8.2f}{v['decoy']:>7.2f}" + "".join(f"{v['by_delay'][str(d)]:>7.2f}" for d in DELAYS))
    print("detection (recall/precision) by cue scale x base noise")
    for r in result["detection"]["rows"]:
        print(f"  {r['detector']:<11} scale {r['cue_scale']:<5} noise {r['base_noise']:<4} R {r['recall']:.2f} P {r['precision']:.2f}")


if __name__ == "__main__":
    main()
