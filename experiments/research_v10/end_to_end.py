"""v10 end to end: do routed memories give a model better ANSWERS on a noisy stream?

After ingesting a v10.1 timeline (facts, typo'd and clean reminders, genuine and one-character updates, untrusted forwarded updates, chatter), the
model is asked for the current value of 6 sampled keys. Conditions (same model, same decoding): ``transcript`` (the whole conversation),
``rag_overlap`` (the 3 lines most similar to the question), ``nsa_hybrid`` (hybrid extractor, no routing), ``nsa_hybrid_routed`` (same, with routing).
Truth is the final trusted value. Scores are lenient exact match on the value token; ``typo_answered`` counts answers equal to a typo seen in the stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from dataclasses import replace

from nsa import NSA, RuntimeConfig
from nsa.memory.extract import HybridFactExtractor
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.grading import lenient_correct
from experiments.research_common.memory_guard import prepare_model
from experiments.research_common.stats import mcnemar_exact, wilson
from experiments.research_v9.natural_memory_benchmark import INSTRUCTION, UNTRUSTED_SOURCE, chunk_events, rag_lines
from experiments.research_v10.gated_memory_benchmark import make_stream

CONDITIONS = ("transcript", "rag_overlap", "nsa_hybrid", "nsa_hybrid_routed", "nsa_routed_fallback")
DECODING = {"temperature": 0.0, "max_tokens": 32, "num_ctx": 8192}


def question(key: str) -> str:
    return f"What is the {key}? {INSTRUCTION}"


def run_episode(backend, seed: int, phrasing: str, n_questions: int = 6) -> list[dict]:
    events, truth, typos = make_stream(seed, phrasing)
    keys = Random(seed).sample(sorted(truth), n_questions)
    agents = {}
    for cond, route in (("nsa_hybrid", False), ("nsa_hybrid_routed", True)):
        a = NSA(backend, config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False,
                                              untrusted_sources=(UNTRUSTED_SOURCE,), route_noisy_updates=route),
                extractor=HybridFactExtractor(backend))
        for text, source in chunk_events(events):
            a.observe(text, source=source)
        agents[cond] = a
    rows = []
    for key in keys:
        q = question(key)
        for cond in CONDITIONS:
            if cond == "transcript":
                text = backend.generate("\n".join(t for t, _ in events) + "\n\n" + q).text
            elif cond == "rag_overlap":
                text = backend.generate("RETRIEVED_LINES=\n" + "\n".join(rag_lines(events, key)) + "\n\n" + q).text
            elif cond == "nsa_routed_fallback":
                # structured memory when the question's key is fully covered by a stored key; otherwise the raw lines most similar to the question
                a = agents["nsa_hybrid_routed"]
                saved = a.config
                a.config = replace(saved, require_full_key_match=True)
                try:
                    keys_found = a.memory_keys_for(key)
                finally:
                    a.config = saved
                if keys_found:
                    prompt = f"RETRIEVED_MEMORY=\n{a.memory.render(keys_found)}\n\n{q}"
                else:
                    prompt = "RETRIEVED_LINES=\n" + "\n".join(rag_lines(events, key)) + "\n\n" + q
                text = backend.generate(prompt).text
            else:
                a = agents[cond]
                keys_found = a.memory_keys_for(key)
                ctx = a.memory.render(keys_found) if keys_found else "NO_RELEVANT_MEMORY"
                text = backend.generate(f"RETRIEVED_MEMORY=\n{ctx}\n\n{q}").text
            rows.append({"seed": seed, "phrasing": phrasing, "key": key, "condition": cond, "expected": truth[key],
                         "correct": lenient_correct(text, truth[key]),
                         "typo_answered": any(t.lower() in text.lower() for t in typos), "response": text})
    return rows


def run(args) -> dict:
    prepare_model(None, args.model, DECODING["num_ctx"])
    backend = FixedDecodingBackend(OllamaInferenceBackend(model=args.model, timeout_sec=300), **DECODING)
    rows = [r for s in args.seeds for ph in ("train", "heldout") for r in run_episode(backend, s, ph)]
    summary = {}
    for ph in ("train", "heldout"):
        for c in CONDITIONS:
            rs = [r for r in rows if r["phrasing"] == ph and r["condition"] == c]
            k = sum(r["correct"] for r in rs)
            summary[f"{ph}|{c}"] = {"n": len(rs), "accuracy": k / len(rs), "accuracy_ci95": list(wilson(k, len(rs))),
                                    "typo_answered": sum(r["typo_answered"] for r in rs) / len(rs)}
    pairs = {}
    for ph in ("train", "heldout"):
        for a, b in (("nsa_hybrid_routed", "nsa_hybrid"), ("nsa_hybrid_routed", "transcript"), ("nsa_hybrid_routed", "rag_overlap"),
                     ("nsa_routed_fallback", "nsa_hybrid_routed"), ("nsa_routed_fallback", "transcript"), ("nsa_routed_fallback", "rag_overlap")):
            da = {(r["seed"], r["key"]): r["correct"] for r in rows if r["phrasing"] == ph and r["condition"] == a}
            db = {(r["seed"], r["key"]): r["correct"] for r in rows if r["phrasing"] == ph and r["condition"] == b}
            pairs[f"{ph}|{a} vs {b}"] = mcnemar_exact(da, db)
    return {"experiment": "v10_end_to_end", "model": args.model, "seeds": args.seeds, "decoding": DECODING, "summary": summary,
            "paired_tests": pairs, "rows": rows,
            "boundary": "Synthetic noisy conversations, opaque values, a 1.5B model; 6 sampled questions per episode."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8)))
    p.add_argument("--out", default="results/research-v10.1-end-to-end-1.5b.json")
    args = p.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for k, v in result["summary"].items():
        print(f"{k:<28} acc {v['accuracy']:.2f} [{v['accuracy_ci95'][0]:.2f},{v['accuracy_ci95'][1]:.2f}]  typo-answered {v['typo_answered']:.2f}  n={v['n']}")
    for k, v in result["paired_tests"].items():
        print(f"{k:<52} only_first={v['only_first']} only_second={v['only_second']} p={v['p']:.3f}")


if __name__ == "__main__":
    main()
