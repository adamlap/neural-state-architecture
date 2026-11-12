"""First LLM wrapper (v0): a fixed-size familiarity memory decides which events deserve an LLM call or a memory write.

Design constraint from earlier NSA results: serialised state pasted into a prompt does nothing; state that CONTROLS something helps. So the
organic module does not talk to the LLM; it gates it. ``NoveltyGate`` keeps an autoassociative linear memory W (D x D floats, fixed) trained online
with the delta rule on hashed character-trigram cues of each event. The novelty of an event is the part of its cue the memory cannot reconstruct,
``||x - W x||``, and the memory then learns the event. Familiar events (repeats, paraphrases, near-duplicates) reconstruct well and are skipped;
genuinely new content does not. A near-duplicate with a typo is *familiar* to a distributed memory and *novel* to an exact set; that is the point.

Benchmark (no LLM is called; the question is how well the gate ranks events by importance, and what a call budget buys):
a synthetic conversation stream of facts about (entity, attribute) with opaque values, reminders (same value, sometimes with a typo or in a different
phrasing), updates (new value) and chatter. Important = first mention of a key, or an update to a different value. Scores, per detector:
ROC AUC for importance, and recall / false-call rate when only 20 % of events may trigger a call. Detectors: the field gate (fixed size), exact
token-set novelty, nearest-neighbour novelty over every stored line (grows with the stream), and random.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from random import Random

import numpy as np

DIM = 256
ENTITIES = ("staging", "billing", "payments", "auth", "search", "reports", "gateway", "mailer", "scheduler", "uploads")
ATTRS = ("database port", "admin email", "api region", "timeout seconds", "cache size", "queue name", "log level", "owner team")
FACT_T = ("The {a} for {e} is {v}.", "FYI: {e} {a} = {v}", "For {e}, use {v} as the {a}.", "{e}'s {a} is {v}, in case you need it.")
UPDATE_T = ("Update: the {a} for {e} is now {v}.", "We changed {e}'s {a} to {v}.", "Heads-up: {e} {a} got swapped to {v}.")
REMIND_T = ("Reminder: {e} {a} is still {v}.", "just to confirm, the {a} for {e} is {v}", "{e} {a}: {v} (unchanged)")
CHATTER = ("ok thanks, will look at it later", "lunch at noon?", "can someone check the build?", "sounds good", "I'll be offline tomorrow",
           "great, merging now", "who is on call this week?", "see you in the standup", "did the deploy finish?", "nice work everyone")
DETECTORS = ("field", "token_set", "knn", "random")


def cue(text: str, dim: int = DIM) -> np.ndarray:
    """Signed hashed character-trigram vector (similar strings -> similar cues), unit norm."""
    v = np.zeros(dim, dtype=np.float32)
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = int.from_bytes(hashlib.blake2b(t[i:i + 3].encode(), digest_size=8).digest(), "little")
        v[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class NoveltyGate:
    """Fixed-size autoassociative familiarity memory. ``score`` then learn; ``floats`` never grows."""

    def __init__(self, dim: int = DIM, eta: float = 0.5) -> None:
        self.W = np.zeros((dim, dim), dtype=np.float32)
        self.eta = eta
        self.floats = dim * dim

    def novelty(self, x: np.ndarray) -> float:
        return float(np.linalg.norm(x - self.W @ x))

    def learn(self, x: np.ndarray) -> None:
        self.W += self.eta * np.outer(x - self.W @ x, x)

    def observe(self, text: str) -> float:
        x = cue(text, self.W.shape[0])
        score = self.novelty(x)
        self.learn(x)
        return score

    def should_act(self, score: float, threshold: float) -> bool:
        return score >= threshold


def typo(value: str, rng: Random) -> str:
    i = rng.randrange(len(value))
    return value[:i] + rng.choice("abcdefghjkmnpqrstuvwxyz23456789") + value[i + 1:]


def stream(seed: int, n: int):
    """Return ``[(text, important)]`` for ``n`` lines."""
    rng = Random(seed)
    token = lambda: "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(8))
    known: dict[tuple[str, str], str] = {}
    out = []
    while len(out) < n:
        r = rng.random()
        e, a = rng.choice(ENTITIES), rng.choice(ATTRS)
        if r < 0.30 or not known and r < 0.8:
            if (e, a) in known:
                continue
            known[(e, a)] = token()
            out.append((rng.choice(FACT_T).format(e=e, a=a, v=known[(e, a)]), True))
        elif r < 0.40 and known:
            (e, a), v = rng.choice(list(known.items()))
            known[(e, a)] = token()
            out.append((rng.choice(UPDATE_T).format(e=e, a=a, v=known[(e, a)]), True))
        elif r < 0.75 and known:
            (e, a), v = rng.choice(list(known.items()))
            shown = typo(v, rng) if rng.random() < 0.4 else v
            out.append((rng.choice(REMIND_T).format(e=e, a=a, v=shown), False))
        else:
            out.append((rng.choice(CHATTER), False))
    return out


def auc(scores: list[float], labels: list[bool]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return 0.5
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def scores_for(detector: str, lines: list[str], seed: int) -> tuple[list[float], int]:
    if detector == "field":
        g = NoveltyGate()
        return [g.observe(t) for t in lines], g.floats
    if detector == "random":
        rng = Random(seed + 17)
        return [rng.random() for _ in lines], 0
    if detector == "token_set":
        seen, out = set(), []
        for t in lines:
            toks = t.lower().replace(",", " ").replace(".", " ").replace(":", " ").split()
            out.append(sum(tok not in seen for tok in toks) / max(1, len(toks)))
            seen.update(toks)
        return out, len(seen)
    stored: list[np.ndarray] = []
    out = []
    for t in lines:
        x = cue(t)
        out.append(1.0 - max((float(x @ s) for s in stored), default=0.0))
        stored.append(x)
    return out, len(stored) * DIM


def call_budget(scores: list[float], labels: list[bool], rate: float) -> dict:
    k = max(1, int(round(len(scores) * rate)))
    top = set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])
    imp = [i for i, l in enumerate(labels) if l]
    caught = sum(i in top for i in imp)
    return {"recall": caught / len(imp) if imp else 0.0, "false_call_rate": (k - caught) / max(1, len(scores) - len(imp))}


def run(*, seeds=tuple(range(10)), lengths=(300, 2000), detectors=DETECTORS, rate=0.2) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    rows = []
    for n in lengths:
        for det in detectors:
            rs = []
            for s in seeds:
                data = stream(s, n)
                sc, floats = scores_for(det, [t for t, _ in data], s)
                labels = [l for _, l in data]
                rs.append({"auc": auc(sc, labels), **call_budget(sc, labels, rate), "floats": floats, "important_share": sum(labels) / n})
            rows.append({"length": n, "detector": det, "floats": rs[-1]["floats"],
                         "auc": statistics.fmean(r["auc"] for r in rs), "auc_min": min(r["auc"] for r in rs),
                         "recall_at_budget": statistics.fmean(r["recall"] for r in rs),
                         "false_call_rate": statistics.fmean(r["false_call_rate"] for r in rs),
                         "important_share": statistics.fmean(r["important_share"] for r in rs)})
    return {"experiment": "organic_novelty_gate_v1", "seeds": list(seeds), "call_rate": rate, "rows": rows,
            "boundary": "Synthetic conversation, opaque values, hashed trigram cues; no LLM is called - this scores the gate's ranking of events."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/organic_novelty_gate_v1.json")
    args = p.parse_args()
    result = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'length':>6} {'detector':<10}{'floats':>9}{'AUC':>7}{'worst':>7}{'recall@20%':>12}{'false-call':>12}")
    for r in result["rows"]:
        print(f"{r['length']:>6} {r['detector']:<10}{r['floats']:>9}{r['auc']:>7.3f}{r['auc_min']:>7.3f}{r['recall_at_budget']:>12.2f}{r['false_call_rate']:>12.2f}")


if __name__ == "__main__":
    main()
