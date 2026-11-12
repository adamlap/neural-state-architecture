"""The "dog test": learning cue -> outcome associations from continuous multi-channel streams under a tiny memory.

No tokens, no exact recall. A world emits three continuous sensory channels (8 dims each) with smooth noise. Cues are
prototype patterns across channels; some are followed (after a delay) by a rewarding (+1) or aversive (-1) outcome,
most are decoys followed by nothing. The agent must learn to predict the outcome at the end of the cue and is scored on
behaviour, not storage.

Organic agent (all parts are measured by ablation):
* trailing field: leaky traces of the input at several time constants, so a cue is still present in the field when the
  delayed outcome arrives (the trace-conditioning mechanism; no explicit credit-assignment buffer of raw inputs);
* separation: fixed random expansion + k-winners-take-all of the field state, a sparse code;
* memory: ONE fixed vector M over the sparse code (prediction = M . code); nothing else is stored per event;
* surprise-gated writes: delta rule applied only when |outcome - prediction| exceeds a threshold;
* replay during idle time from a small buffer of the most surprising events, re-applied with the same delta rule.

Baselines under the *same float budget*: k-nearest-neighbour over stored (field state, outcome) entries with FIFO,
random-reservoir and surprise-gated retention, and an unbounded kNN as a perfect-recall reference.

Scores: sign accuracy on held-out noisy cue variants (decoys must be predicted ~0), graded response on morphs between a
rewarding and an aversive cue (generalisation), accuracy after a long run of decoy-only events (retention), and the
learning curve. Boundary: a toy world with synthetic prototypes; this measures learning and selectivity, not awareness.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODALITIES, DIM = 3, 8
X = MODALITIES * DIM
CUE_STEPS, DELAY, GAP = 4, 3, (6, 12)
LAMBDA_MID = 0.8
N_VALENCED, N_DECOY = 6, 12
BASE_NOISE, CUE_NOISE, TEST_NOISE = 0.2, 0.3, 0.6
SPARSE_DIM, SPARSITY = 1024, 0.05
THETA, ETA = 0.15, 0.5
REPLAY_BUFFER, REPLAY_PASSES = 32, 3
AGENTS = ("organic", "organic_no_replay", "organic_write_all", "knn_fifo", "knn_random", "knn_surprise", "knn_all", "none")


@dataclass
class World:
    prototypes: np.ndarray      # (types, X)
    valence: np.ndarray         # (types,)
    rng: np.random.Generator

    @classmethod
    def make(cls, seed: int) -> "World":
        rng = np.random.default_rng(seed)
        protos = rng.standard_normal((N_VALENCED + N_DECOY, X)).astype(np.float32)
        protos /= np.linalg.norm(protos, axis=1, keepdims=True) / np.sqrt(X)          # unit RMS per channel
        val = np.zeros(N_VALENCED + N_DECOY, dtype=np.float32)
        val[: N_VALENCED // 2], val[N_VALENCED // 2: N_VALENCED] = 1.0, -1.0
        return cls(protos, val, rng)

    def cue_sample(self, kind: int, noise: float, scale: float = 1.0) -> np.ndarray:
        return scale * self.prototypes[kind] + noise * self.rng.standard_normal(X).astype(np.float32)


class Field:
    """Trailing neural field: leaky integration of the sensory stream (one mid time constant is used as the key)."""

    def __init__(self) -> None:
        self.z = np.zeros(X, dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        self.z = LAMBDA_MID * self.z + (1.0 - LAMBDA_MID) * x
        return self.z

    def key(self) -> np.ndarray:
        n = np.linalg.norm(self.z)
        return self.z / n if n > 1e-8 else self.z.copy()


def cue_end_key(world: World, kind: int, noise: float, scale: float = 1.0, proto: np.ndarray | None = None) -> np.ndarray:
    f = Field()
    for _ in range(6):
        f.step(BASE_NOISE * world.rng.standard_normal(X).astype(np.float32))
    for _ in range(CUE_STEPS):
        base = world.prototypes[kind] if proto is None else proto
        f.step(scale * base + noise * world.rng.standard_normal(X).astype(np.float32))
    return f.key()


class Separator:
    def __init__(self, seed: int) -> None:
        self.w = np.random.default_rng(seed + 9001).standard_normal((SPARSE_DIM, X)).astype(np.float32)
        self.k = int(round(SPARSE_DIM * SPARSITY))

    def __call__(self, key: np.ndarray) -> np.ndarray:
        drive = self.w @ key
        code = np.zeros(SPARSE_DIM, dtype=np.float32)
        code[np.argpartition(drive, -self.k)[-self.k:]] = 1.0
        return code / np.sqrt(self.k)


class Organic:
    def __init__(self, seed: int, *, theta: float = THETA, replay: bool = True) -> None:
        self.sep, self.M, self.theta, self.replay = Separator(seed), np.zeros(SPARSE_DIM, dtype=np.float32), theta, replay
        self.buffer: list[tuple[float, np.ndarray, float]] = []       # (surprise, key, outcome)
        self.floats = SPARSE_DIM + (REPLAY_BUFFER * (X + 1) if replay else 0)

    def predict(self, key: np.ndarray) -> float:
        return float(self.M @ self.sep(key))

    def _delta(self, key: np.ndarray, target: float) -> float:
        code = self.sep(key)
        err = target - float(self.M @ code)
        self.M += ETA * err * code / float(code @ code)
        return err

    def observe(self, key: np.ndarray, outcome: float, predicted: float) -> None:
        err = abs(outcome - predicted)
        if err <= self.theta:
            return
        self._delta(key, outcome)
        if self.replay:
            self.buffer.append((err, key, outcome))
            self.buffer = sorted(self.buffer, key=lambda t: -t[0])[:REPLAY_BUFFER]      # keep the most surprising

    def idle(self) -> None:
        if not self.replay:
            return
        for _ in range(REPLAY_PASSES):
            self.buffer = [(abs(o - self.predict(k)), k, o) for _, k, o in self.buffer]
            for _, k, o in sorted(self.buffer, key=lambda t: -t[0]):
                self._delta(k, o)


class Knn:
    def __init__(self, policy: str, budget_floats: int, seed: int) -> None:
        self.policy, self.cap = policy, (10 ** 9 if policy == "all" else max(1, budget_floats // (X + 1)))
        self.keys: list[np.ndarray] = []
        self.out: list[float] = []
        self.surprise: list[float] = []
        self.rng = np.random.default_rng(seed + 31337)
        self.seen = 0
        self.floats = self.cap * (X + 1) if policy != "all" else 0

    def predict(self, key: np.ndarray) -> float:
        if not self.keys:
            return 0.0
        sims = np.stack(self.keys) @ key
        idx = np.argsort(-sims)[:3]
        w = np.maximum(sims[idx], 1e-6)
        return float(np.dot(w, np.array(self.out)[idx]) / w.sum())

    def observe(self, key: np.ndarray, outcome: float, predicted: float) -> None:
        err = abs(outcome - predicted)
        self.seen += 1
        if self.policy == "surprise" and err <= THETA:
            return
        if len(self.keys) < self.cap:
            self.keys.append(key); self.out.append(outcome); self.surprise.append(err); return
        if self.policy in ("fifo",):
            self.keys.pop(0); self.out.pop(0); self.surprise.pop(0)
            self.keys.append(key); self.out.append(outcome); self.surprise.append(err)
        elif self.policy == "random":                       # reservoir sampling
            j = int(self.rng.integers(0, self.seen))
            if j < self.cap:
                self.keys[j], self.out[j], self.surprise[j] = key, outcome, err
        elif self.policy == "surprise":                     # evict the least surprising entry
            j = int(np.argmin(self.surprise))
            if err > self.surprise[j]:
                self.keys[j], self.out[j], self.surprise[j] = key, outcome, err

    def idle(self) -> None:
        pass


class NoMemory:
    floats = 0

    def predict(self, key): return 0.0
    def observe(self, key, outcome, predicted): pass
    def idle(self): pass


def make_agent(name: str, seed: int, budget: int):
    if name == "organic":
        return Organic(seed)
    if name == "organic_no_replay":
        return Organic(seed, replay=False)
    if name == "organic_write_all":
        return Organic(seed, theta=-1.0)
    if name == "none":
        return NoMemory()
    return Knn(name.split("_", 1)[1], budget, seed)


def evaluate(agent, world: World) -> dict:
    """Held-out noisy variants, decoys, and morphs between a rewarding and an aversive cue."""
    val_ok, dec_ok = [], []
    for kind in range(N_VALENCED + N_DECOY):
        for _ in range(8):
            p = agent.predict(cue_end_key(world, kind, TEST_NOISE, scale=float(world.rng.uniform(0.7, 1.3))))
            if kind < N_VALENCED:
                val_ok.append(float(np.sign(p) == world.valence[kind] and abs(p) > 0.25))
            else:
                dec_ok.append(float(abs(p) < 0.5))
    weights = np.linspace(0.0, 1.0, 5)
    pa, pb = world.prototypes[0], world.prototypes[N_VALENCED // 2]                 # one rewarding, one aversive
    morph = [np.mean([agent.predict(cue_end_key(world, 0, CUE_NOISE, proto=w * pb + (1 - w) * pa)) for _ in range(6)]) for w in weights]
    graded = float(np.corrcoef(weights, morph)[0, 1]) if np.std(morph) > 1e-6 else 0.0
    return {"valenced_accuracy": statistics.fmean(val_ok), "decoy_accuracy": statistics.fmean(dec_ok), "morph_monotone_corr": -graded,
            "morph_response": [float(m) for m in morph]}


def train_events(agent, world: World, n_events: int, *, decoy_only: bool = False) -> None:
    for _ in range(n_events):
        kind = int(world.rng.integers(N_VALENCED, N_VALENCED + N_DECOY)) if decoy_only else (
            int(world.rng.integers(0, N_VALENCED)) if world.rng.random() < 1 / 3 else int(world.rng.integers(N_VALENCED, N_VALENCED + N_DECOY)))
        field = Field()
        for _ in range(int(world.rng.integers(*GAP))):
            field.step(BASE_NOISE * world.rng.standard_normal(X).astype(np.float32))
        for _ in range(CUE_STEPS):
            field.step(world.cue_sample(kind, CUE_NOISE))
        key = field.key()
        predicted = agent.predict(key)
        for _ in range(DELAY):
            field.step(BASE_NOISE * world.rng.standard_normal(X).astype(np.float32))
        agent.observe(key, float(world.valence[kind]), predicted)           # outcome (0 if nothing follows) after the delay
        agent.idle()


def run_one(name: str, seed: int, *, events=(25, 50, 100, 200, 400), distractor_events: int = 400, budget_floats: int | None = None) -> dict:
    budget = budget_floats if budget_floats is not None else Organic(seed).floats
    world, agent = World.make(seed), make_agent(name, seed, budget)
    curve, done = [], 0
    for target in events:
        train_events(agent, world, target - done)
        done = target
        curve.append({"events": target, **{k: v for k, v in evaluate(agent, world).items() if k != "morph_response"}})
    final = evaluate(agent, world)
    train_events(agent, world, distractor_events, decoy_only=True)
    after = evaluate(agent, world)
    return {"agent": name, "seed": seed, "floats": agent.floats, "curve": curve, "final": final, "after_distractors": after}


def run(*, seeds=tuple(range(10)), agents=AGENTS, **kw) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    rows = [run_one(a, s, **kw) for a in agents for s in seeds]
    agg = {}
    for a in agents:
        rs = [r for r in rows if r["agent"] == a]
        agg[a] = {"floats": rs[0]["floats"],
                  "valenced_accuracy": statistics.fmean(r["final"]["valenced_accuracy"] for r in rs),
                  "decoy_accuracy": statistics.fmean(r["final"]["decoy_accuracy"] for r in rs),
                  "morph_monotone_corr": statistics.fmean(r["final"]["morph_monotone_corr"] for r in rs),
                  "valenced_after_distractors": statistics.fmean(r["after_distractors"]["valenced_accuracy"] for r in rs),
                  "learning_curve": [statistics.fmean(r["curve"][i]["valenced_accuracy"] for r in rs) for i in range(len(rs[0]["curve"]))],
                  "curve_events": [c["events"] for c in rs[0]["curve"]]}
    return {"experiment": "organic_dog_test_v1", "seeds": list(seeds), "aggregate": agg, "runs": rows,
            "boundary": "Toy world with synthetic prototypes; measures learning, generalisation and selectivity under a float budget, not awareness."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_dog_test_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'agent':<20}{'floats':>8}{'valenced':>10}{'decoy':>8}{'morph r':>9}{'after distr.':>14}  learning curve (valenced acc @ events)")
    for a, v in result["aggregate"].items():
        print(f"{a:<20}{v['floats']:>8}{v['valenced_accuracy']:>10.2f}{v['decoy_accuracy']:>8.2f}{v['morph_monotone_corr']:>9.2f}{v['valenced_after_distractors']:>14.2f}  "
              + " ".join(f"{x:.2f}" for x in v["learning_curve"]))


if __name__ == "__main__":
    main()
