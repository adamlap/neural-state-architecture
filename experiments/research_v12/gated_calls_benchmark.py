"""NSA Research Benchmark v12: gating model calls inside the runtime (abstain + reuse) on a re-asked question stream.

After ingesting a conversation (train phrasing, rule extraction, no model) a stream of questions is asked through ``NSARuntime.step(..., auto_memory=True)``.
Keys are asked repeatedly (exact repeats, one-character typos, and paraphrases), 15 % of questions are about an entity that does not exist, and genuine updates are
interleaved so that an earlier answer can become stale. Conditions: ``always_call`` (every question goes to the model with its retrieved memory) and ``gated``
(``abstain_text="VALUE=UNKNOWN"`` and ``reuse_familiar_answers=True``).

Measured per condition: answer accuracy against the truth AT THAT TIME, hallucination on unknown entities, model calls, answers reused / abstained, and *stale reuse* (a reused
answer that is not the current truth), which must be 0 because any write to a retrieved key invalidates its cached answers. Wall time is the sum of per-question latencies.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from random import Random

from nsa import NSA, RuntimeConfig
from nsa.memory.extract import RuleFactExtractor
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.grading import lenient_correct
from experiments.research_common.memory_guard import prepare_model
from experiments.research_common.stats import wilson
from experiments.research_v9.natural_memory_benchmark import ATTRS, ENTITIES, FACT_T, INSTRUCTION, UPDATE_T, _token

VERSION = "12.0"
CONDITIONS = ("always_call", "gated")
DECODING = {"temperature": 0.0, "max_tokens": 32, "num_ctx": 8192}
PARAPHRASES = ("Tell me the {attr} of {ent}.", "Could you give the {attr} for {ent}?", "{ent}: what's the {attr} again?")


def corrupt(text: str, rng: Random) -> str:
    """One-character typo inside a word of at least 5 letters."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    j = rng.randrange(1, len(words[i]) - 1)
    words[i] = words[i][:j] + words[i][j + 1:]
    return " ".join(words)


def make_session(seed: int, n_facts: int = 12, n_questions: int = 60, n_updates: int = 6, unknown_share: float = 0.15):
    """Return ``(facts, events)``: ``facts`` is a list of ``(line, key_text)``; events are ``("ask", question, entity, attr)`` / ``("update", line, entity, attr, value)``."""
    rng = Random(seed)
    combos = [(e, a) for e in ENTITIES for a in ATTRS]
    rng.shuffle(combos)
    chosen = combos[:n_facts]
    values = {c: _token(rng) for c in chosen}
    facts = [(rng.choice(FACT_T["train"]).format(ent=e, attr=a, val=values[(e, a)]), (e, a)) for e, a in chosen]
    events = []
    update_slots = set(rng.sample(range(n_questions), n_updates))
    for i in range(n_questions):
        if i in update_slots:
            e, a = rng.choice(chosen)
            v = _token(rng)
            events.append(("update", rng.choice(UPDATE_T["train"]).format(ent=e, attr=a, val=v), e, a, v))
        if rng.random() < unknown_share:
            e = rng.choice([x for x in ENTITIES if all(x != c[0] for c in chosen)] or ["zebra"])
            a = rng.choice(ATTRS)
            events.append(("ask", f"What is the {a} for {e}?", e, a))
            continue
        e, a = rng.choice(chosen)
        r = rng.random()
        q = f"What is the {a} for {e}?" if r < 0.5 else (corrupt(f"What is the {a} for {e}?", rng) if r < 0.75 else rng.choice(PARAPHRASES).format(attr=a, ent=e))
        events.append(("ask", q, e, a))
    return facts, values, events


def run_session(backend, seed: int, gated: bool) -> dict:
    facts, values, events = make_session(seed)
    cfg = RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False,
                        abstain_text="VALUE=UNKNOWN" if gated else None, reuse_familiar_answers=gated, require_full_key_match=True)
    agent = NSA(backend, config=cfg, extractor=RuleFactExtractor())
    for line, _ in facts:
        agent.observe(line, source="chat")
    truth = dict(values)
    version = {c: 0 for c in values}               # bumped by every genuine update
    answered_at: dict = {}                         # (entity, attribute) -> truth version when the model last produced an answer
    rows = []
    for ev in events:
        if ev[0] == "update":
            agent.observe(ev[1], source="chat")
            truth[(ev[2], ev[3])] = ev[4]
            version[(ev[2], ev[3])] += 1
            continue
        _, q, e, a = ev
        before = dict(agent.call_stats)
        start = time.perf_counter()
        res = agent.step(f"{q} {INSTRUCTION}", auto_memory=True, remember_prompt=False)
        latency = time.perf_counter() - start
        known = (e, a) in truth
        reused = agent.call_stats["reused"] > before["reused"]
        abstained = agent.call_stats["abstained"] > before["abstained"]
        correct = (("UNKNOWN" in res.text.upper()) if not known else lenient_correct(res.text, truth[(e, a)]))
        # stale reuse = an answer reused after the truth for that key changed; reused_wrong = a reused answer that was wrong at its origin
        stale = bool(reused and known and answered_at.get((e, a), version[(e, a)]) != version[(e, a)])
        if known and not reused and not abstained:
            answered_at[(e, a)] = version[(e, a)]
        rows.append({"seed": seed, "condition": "gated" if gated else "always_call", "known": known, "correct": bool(correct), "reused": reused,
                     "abstained": abstained, "stale_reuse": stale, "reused_wrong": bool(reused and known and not correct),
                     "abstained_known": bool(abstained and known),
                     "hallucinated": (not known) and "UNKNOWN" not in res.text.upper(), "latency": latency, "question": q, "response": res.text})
    return {"rows": rows, "call_stats": agent.call_stats}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for c in CONDITIONS:
        rs = [r for r in rows if r["condition"] == c]
        known = [r for r in rs if r["known"]]
        unknown = [r for r in rs if not r["known"]]
        k = sum(r["correct"] for r in known)
        out[c] = {"questions": len(rs), "model_calls": sum(not (r["reused"] or r["abstained"]) for r in rs),
                  "reused": sum(r["reused"] for r in rs), "abstained": sum(r["abstained"] for r in rs),
                  "known_accuracy": k / len(known) if known else None, "known_ci95": list(wilson(k, len(known))) if known else None,
                  "hallucination_rate": sum(r["hallucinated"] for r in unknown) / len(unknown) if unknown else None,
                  "stale_reuse": sum(r["stale_reuse"] for r in rs), "reused_wrong": sum(r["reused_wrong"] for r in rs),
                  "abstained_on_known": sum(r["abstained_known"] for r in rs), "mean_latency_s": statistics.fmean(r["latency"] for r in rs),
                  "total_time_s": sum(r["latency"] for r in rs)}
    return out


def run(args) -> dict:
    prepare_model(args.ollama_url, args.model, DECODING["num_ctx"])
    inner = OllamaInferenceBackend(model=args.model, base_url=args.ollama_url, timeout_sec=args.timeout) if args.ollama_url \
        else OllamaInferenceBackend(model=args.model, timeout_sec=args.timeout)
    backend = FixedDecodingBackend(inner, **DECODING)
    rows = []
    for seed in args.seeds:
        for gated in (False, True):
            rows += run_session(backend, seed, gated)["rows"]
            print(f"seed {seed} gated={gated} done", flush=True)
    return {"experiment": "v12_gated_calls", "version": VERSION, "model": args.model, "seeds": args.seeds, "summary": summarize(rows), "rows": rows,
            "boundary": "Synthetic session; rule-extracted memory; one small model; reuse keyed on retrieved-key versions plus question trigram cosine >= 0.95."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--out", default="results/research-v12-1.5b.json")
    args = p.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for c, v in result["summary"].items():
        print(f"{c:<12} questions {v['questions']:>4}  model calls {v['model_calls']:>4}  reused {v['reused']:>3}  abstained {v['abstained']:>3}  "
              f"known acc {v['known_accuracy']:.2f}  halluc {v['hallucination_rate']:.2f}  stale reuse {v['stale_reuse']}  reused-wrong {v['reused_wrong']}  "
              f"abstained-on-known {v['abstained_on_known']}  total {v['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
