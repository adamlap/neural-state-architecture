"""NSA Research Benchmark v10: selective storage with a familiarity gate on noisy streams.

Extends the v9 conversations (facts, genuine updates, an untrusted forwarded update, chatter) with *reminders*: lines that restate an earlier
fact, half of them with a one-character typo in the value (transcription noise). An extractor that writes everything it parses will record the
typo as a new value; a familiarity gate (``nsa.organic.NoveltyGate``) should treat a near-duplicate as familiar and skip it, while still passing
genuine updates (a whole new value in new wording).

10.1 (timeline stream): adds ``route`` (``RuntimeConfig.route_noisy_updates``: ignore restatements, hold a near-duplicate value without an update cue until it is
seen twice) and *near genuine updates*: real changes that differ from the old value by one character and carry an explicit update cue ("now",
"changed to"), which a naive "close value = noise" rule would wrongly reject. 10.0 numbers used an extractor that did not parse "is still X" and let
lead-ins such as "just to confirm," leak into the key; both were fixed before 10.1 and the 10.0 file is superseded.

Conditions: the same extractor with and without the gate, at several thresholds (``NSARuntime`` ``novelty_write_threshold``), with routing off/on. Everything is
measured on the canonical memory after ingest, with no answer model: ``all_keys_correct`` (fraction of keys whose current value is the true
final value), ``target_correct`` (the queried key), ``corrupted`` (fraction of keys whose current value is a typo of the truth), memory writes,
and lines passed to the extractor (the quantity an LLM extractor pays for).

Boundary: synthetic conversations with opaque values; the gate is lexical, so a genuine update whose wording closely repeats earlier text can be
dropped (measured as lower ``all_keys_correct`` at high thresholds).
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

from nsa import NSA, RuntimeConfig
from nsa.memory.extract import HybridFactExtractor, LLMFactExtractor, RuleFactExtractor, normalize_key
from nsa.organic import NoveltyGate

from experiments.research_v9.natural_memory_benchmark import (
    ATTRS, CHATTER, ENTITIES, FACT_T, UNTRUSTED_SOURCE, UPDATE_T, _token, chunk_events,
)

VERSION = "10.1"
THRESHOLDS = (None, 0.3, 0.45, 0.6, 0.75)
REMIND_T = ("Reminder: {e} {a} is still {v}.", "just to confirm, the {a} for {e} is {v}", "{e} {a}: {v} (unchanged)")


def typo(value: str, rng: Random) -> str:
    i = rng.randrange(len(value))
    return value[:i] + rng.choice([c for c in "abcdefghjkmnpqrstuvwxyz23456789" if c != value[i]]) + value[i + 1:]


def make_stream(seed: int, phrasing: str, n_facts: int = 20, reminders: int = 20, updates: int = 4, poisoned: int = 2, near_updates: int = 2):
    """Return ``(events, truth, typo_values)``; events are ``(text, source)``; truth maps normalised key -> final trusted value.

    The stream is a timeline: each reminder restates the value that is *current at that point* (half with a one-character typo), a genuine update
    changes it (``near_updates`` of them by a single character, all with an explicit update cue), and untrusted forwarded updates never change the truth.
    (An earlier version generated reminders from a snapshot and shuffled them past later updates, which produced stale reminders whose truth is ambiguous.)
    """
    rng = Random(seed * 7919 + (0 if phrasing == "train" else 1))
    combos = [(e, a) for e in ENTITIES for a in ATTRS]
    rng.shuffle(combos)
    chosen = combos[:n_facts]
    value = {c: _token(rng) for c in chosen}
    events = [(rng.choice(FACT_T[phrasing]).format(ent=e, attr=a, val=value[(e, a)]), "chat") for e, a in chosen]
    typos: set[str] = set()
    slots = ["reminder"] * reminders + ["update"] * (updates - near_updates) + ["near_update"] * near_updates + ["poisoned"] * poisoned + ["chatter"] * 10
    rng.shuffle(slots)
    for slot in slots:
        c = rng.choice(chosen)
        if slot == "reminder":
            v = typo(value[c], rng) if rng.random() < 0.5 else value[c]
            if v != value[c]:
                typos.add(v)
            events.append((rng.choice(REMIND_T).format(e=c[0], a=c[1], v=v), "chat"))
        elif slot in ("update", "near_update"):
            value[c] = typo(value[c], rng) if slot == "near_update" else _token(rng)
            events.append((rng.choice(UPDATE_T[phrasing]).format(ent=c[0], attr=c[1], val=value[c]), "chat"))
        elif slot == "poisoned":
            events.append((rng.choice(UPDATE_T[phrasing]).format(ent=c[0], attr=c[1], val=_token(rng)), UNTRUSTED_SOURCE))
        else:
            events.append((rng.choice(CHATTER), "chat"))
    return events, {normalize_key(f"{e} {a}"): v for (e, a), v in value.items()}, typos


@dataclass
class Record:
    extractor: str
    threshold: float | None
    route: bool
    phrasing: str
    seed: int
    all_keys_correct: float
    corrupted: float
    memory_writes: int
    lines_total: int
    lines_extracted: int
    llm_calls: int


def run_one(extractor: str, threshold: float | None, phrasing: str, seed: int, backend=None, route: bool = False) -> Record:
    events, truth, typos = make_stream(seed, phrasing)
    if extractor == "rules":
        ex = RuleFactExtractor()
    elif extractor == "hybrid":
        ex = HybridFactExtractor(backend)
    else:
        raise ValueError(extractor)
    agent = NSA(backend or _Null(), config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False,
                                                          untrusted_sources=(UNTRUSTED_SOURCE,), novelty_write_threshold=threshold,
                                                          route_noisy_updates=route),
                extractor=ex, novelty_gate=NoveltyGate() if threshold is not None else None)
    for text, source in (chunk_events(events) if extractor == "hybrid" else events):
        agent.observe(text, source=source)
    ok = bad = 0
    for key, v in truth.items():
        cur = agent.memory.current(key)
        got = (cur.content.get("value") if cur else None) or ""
        ok += got.lower() == v.lower()
        bad += got in typos or got.lower() in {t.lower() for t in typos}
    lines_total = sum(len(t.splitlines()) for t, _ in (chunk_events(events) if extractor == "hybrid" else events))
    extracted = agent.gate_stats["extracted"] if threshold is not None else lines_total
    return Record(extractor, threshold, route, phrasing, seed, ok / len(truth), bad / len(truth), len(agent.memory.store.items),
                  lines_total, extracted, getattr(getattr(ex, "llm", ex), "calls", 0))


class _Null:
    model = "null"

    def generate(self, prompt, **kw):
        raise RuntimeError("no model configured")


def summarize(records: list[Record]) -> list[dict]:
    rows = []
    for ex in sorted({r.extractor for r in records}):
        for ph in ("train", "heldout"):
            for th, rt in [(t, r) for t in THRESHOLDS for r in (False, True)]:
                rs = [r for r in records if (r.extractor, r.phrasing, r.threshold, r.route) == (ex, ph, th, rt)]
                if rs:
                    rows.append({"extractor": ex, "phrasing": ph, "threshold": th, "route": rt, "n": len(rs),
                                 "all_keys_correct": statistics.fmean(r.all_keys_correct for r in rs),
                                 "corrupted": statistics.fmean(r.corrupted for r in rs),
                                 "memory_writes": statistics.fmean(r.memory_writes for r in rs),
                                 "lines_extracted": statistics.fmean(r.lines_extracted for r in rs),
                                 "llm_calls": statistics.fmean(r.llm_calls for r in rs)})
    return rows


def run(*, seeds=tuple(range(10)), extractors=("rules",), thresholds=(None, 0.75), routes=(False, True), backend=None) -> dict:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    records = [run_one(ex, th, ph, s, backend, rt) for ex in extractors for ph in ("train", "heldout") for th in thresholds for rt in routes for s in seeds]
    return {"experiment": "gated_memory_v10", "version": VERSION, "seeds": list(seeds), "thresholds": list(thresholds),
            "rows": summarize(records), "runs": [asdict(r) for r in records],
            "boundary": "Synthetic noisy conversations; lexical gate; canonical memory measured after ingest without an answer model."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extractors", nargs="+", default=["rules"])
    p.add_argument("--model", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--out", default="results/research-v10.1-rules.json")
    args = p.parse_args()
    backend = None
    if "hybrid" in args.extractors:
        from experiments.research_common.backends import FixedDecodingBackend
        from experiments.research_common.memory_guard import prepare_model
        from nsa.runtime.inference.ollama import OllamaInferenceBackend
        prepare_model(None, args.model, 8192)
        backend = FixedDecodingBackend(OllamaInferenceBackend(model=args.model, timeout_sec=300), temperature=0.0, max_tokens=512, num_ctx=8192)
    result = run(seeds=tuple(args.seeds), extractors=tuple(args.extractors), backend=backend)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'extractor':<8}{'phrasing':<9}{'thresh':>7}{'route':>6}{'all keys ok':>13}{'corrupted':>11}{'writes':>8}{'lines->extractor':>18}{'LLM calls':>11}")
    for r in result["rows"]:
        print(f"{r['extractor']:<8}{r['phrasing']:<9}{str(r['threshold']):>7}{str(r['route']):>6}{r['all_keys_correct']:>13.2f}{r['corrupted']:>11.2f}{r['memory_writes']:>8.0f}"
              f"{r['lines_extracted']:>18.0f}{r['llm_calls']:>11.1f}")


if __name__ == "__main__":
    main()
