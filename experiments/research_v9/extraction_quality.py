"""Extraction quality in isolation (no answer model): precision/recall of extracted (key, value) pairs.

For each episode the whole conversation (facts, an update, chatter; target + distractor lines) is given to an
extractor in chunks of up to 8 lines. Ground truth is every fact/update line's ``(normalize_key(entity attribute),
value)``. Reported per extractor and phrasing set (``train`` / ``heldout``): recall, precision, extra
(spurious) extractions on chatter, model calls and seconds per episode.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nsa.memory.extract import HybridFactExtractor, LLMFactExtractor, RuleFactExtractor, normalize_key
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.memory_guard import prepare_model
from experiments.research_common.stats import wilson
from experiments.research_v9.natural_memory_benchmark import CHATTER, ENTITIES, ATTRS, FACT_T, UPDATE_T, _token, chunk_events
from random import Random

VERSION = "9.1-extraction"
EXTRACTORS = ("rules", "llm", "llm_fewshot", "hybrid")
PHRASINGS = ("train", "heldout")
DECODING = {"temperature": 0.0, "max_tokens": 512, "num_ctx": 8192}


def make_chunk_episode(seed: int, phrasing: str, lines: int = 10):
    """Return ``(events, truth)``; truth is a set of ``(key, value)`` for every fact/update line."""
    rng = Random(seed * 7727 + PHRASINGS.index(phrasing))
    combos = [(e, a) for e in ENTITIES for a in ATTRS]
    rng.shuffle(combos)
    events, truth = [], set()
    for i in range(lines):
        r = rng.random()
        if r < 0.35:
            events.append((rng.choice(CHATTER), "chat"))
            continue
        ent, attr = combos[i % len(combos)]
        val = _token(rng)
        templates = UPDATE_T[phrasing] if r > 0.8 else FACT_T[phrasing]
        events.append((rng.choice(templates).format(ent=ent, attr=attr, val=val), "chat"))
        truth.add((normalize_key(f"{ent} {attr}"), val.lower()))
    return events, truth


def build_extractor(name: str, backend):
    return {"rules": lambda: RuleFactExtractor(), "llm": lambda: LLMFactExtractor(backend),
            "llm_fewshot": lambda: LLMFactExtractor(backend, few_shot=True),
            "hybrid": lambda: HybridFactExtractor(backend)}[name]()


def score(extractor, events, truth) -> dict:
    start = time.perf_counter()
    got = set()
    for text, _ in chunk_events(events):
        got |= {(f.key, f.value.lower()) for f in extractor.extract(text)}
    secs = time.perf_counter() - start
    hit = got & truth
    return {"truth": len(truth), "extracted": len(got), "hit": len(hit), "spurious": len(got - truth),
            "calls": getattr(extractor, "calls", 0), "seconds": round(secs, 3)}


@dataclass
class Record:
    model: str
    extractor: str
    phrasing: str
    seed: int
    truth: int
    extracted: int
    hit: int
    spurious: int
    calls: int
    seconds: float


def summarize(records: list[Record]) -> dict:
    out = {}
    for ex in EXTRACTORS:
        for p in PHRASINGS:
            rows = [r for r in records if r.extractor == ex and r.phrasing == p]
            if not rows:
                continue
            truth, hit, ext = sum(r.truth for r in rows), sum(r.hit for r in rows), sum(r.extracted for r in rows)
            out[f"{ex}|{p}"] = {
                "episodes": len(rows), "recall": hit / truth if truth else None, "recall_ci95": list(wilson(hit, truth)),
                "precision": hit / ext if ext else None, "spurious_per_episode": sum(r.spurious for r in rows) / len(rows),
                "calls_per_episode": sum(r.calls for r in rows) / len(rows), "seconds_per_episode": sum(r.seconds for r in rows) / len(rows),
            }
    return out


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    records: list[Record] = []
    for model in args.models:
        prepare_model(args.ollama_url, model, DECODING["num_ctx"])
        inner = OllamaInferenceBackend(model=model, base_url=args.ollama_url, timeout_sec=args.timeout) \
            if args.ollama_url else OllamaInferenceBackend(model=model, timeout_sec=args.timeout)
        backend = FixedDecodingBackend(inner, **DECODING)
        for name in EXTRACTORS:
            if name == "rules" and model != args.models[0]:
                continue                                    # model-independent; run once
            for phrasing in PHRASINGS:
                for seed in args.seeds:
                    events, truth = make_chunk_episode(seed, phrasing, args.lines)
                    res = score(build_extractor(name, backend), events, truth)
                    records.append(Record(model if name != "rules" else "none", name, phrasing, seed, **res))
                    print(f"{model} {name} {phrasing} s={seed} hit={res['hit']}/{res['truth']} spurious={res['spurious']} calls={res['calls']}", flush=True)
    (root / "raw.jsonl").write_text("\n".join(json.dumps(asdict(r)) for r in records) + "\n", encoding="utf-8")
    per_model = {m: summarize([r for r in records if r.model in (m, "none")]) for m in args.models}
    manifest = {"benchmark": "NSA extraction quality", "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "models": args.models, "seeds": args.seeds, "lines": args.lines, "decoding": DECODING, "by_model": per_model,
                "scientific_boundary": "Synthetic template conversations with opaque value tokens; held-out templates were written after the rules were fixed."}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(7, 27)))
    p.add_argument("--lines", type=int, default=10)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", default="results/research-v9-extraction")
    p.add_argument("--ollama-url", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
