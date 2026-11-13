"""NSA Research Benchmark v13: raw-line retrieval measured through the real runtime path (``NSARuntime.step``) on long noisy histories.

v10 measured "memory + raw lines" with hand-built prompts on 100-line timelines. Here the agent is built by ``observe`` (rule extractor, routing, signed untrusted
source) over 100 / 1000 / 3000 lines and each question goes through ``step(question, auto_memory=True, remember_prompt=False, retrieval_query=key)``, so the
inverted line index, key retrieval and prompt assembly are the shipped code. ``heldout`` phrasing is the interesting case: the rule extractor cannot read it, so
memory is empty and only the raw lines can answer.

Conditions: ``mem_only`` (retrieve_lines=0), ``mem_lines`` (retrieve_lines=3), ``rag_overlap`` (v9 scan retrieval, 3 lines), ``transcript`` (whole history; skipped
above ``--transcript-token-cap``). Also recorded: wall time per question and per-``lines_for`` latency.
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
from experiments.research_v9.natural_memory_benchmark import INSTRUCTION, UNTRUSTED_SOURCE, rag_lines
from experiments.research_v10.gated_memory_benchmark import make_stream

VERSION = "13.0"
CONDITIONS = ("mem_only", "mem_lines", "rag_overlap", "transcript")
DECODING = {"temperature": 0.0, "max_tokens": 32}
CHARS_PER_TOKEN = 3.3


def long_stream(seed: int, lines: int, phrasing: str):
    body = max(0, lines - 80)
    return make_stream(seed, phrasing, n_facts=80, reminders=int(body * 0.735), updates=max(2, body // 20), poisoned=max(2, body // 25),
                       near_updates=max(2, body // 40), chatter=int(body * 0.15))


def build_agent(events, backend, retrieve_lines: int, window: int | None) -> tuple[NSA, float]:
    a = NSA(backend, config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False, untrusted_sources=(UNTRUSTED_SOURCE,),
                                          route_noisy_updates=True, retrieve_lines=retrieve_lines, provenance_window=window), extractor=RuleFactExtractor())
    t0 = time.perf_counter()
    for text, source in events:
        a.observe(text, source=source)
    ingest = time.perf_counter() - t0
    return a, ingest


def run_episode(backend, seed: int, lines: int, phrasing: str, cap: int, n_questions: int, window: int | None) -> list[dict]:
    events, truth, typos = long_stream(seed, lines, phrasing)
    keys = Random(seed).sample(sorted(truth), n_questions)
    transcript = "\n".join(t for t, _ in events)
    agents, ingest = {}, {}
    for cond, k in (("mem_only", 0), ("mem_lines", 3)):
        a, ingest[cond] = build_agent(events, backend, k, window)
        agents[cond] = a
    rows = []
    for key in keys:
        q = f"What is the {key}? {INSTRUCTION}"
        for cond in CONDITIONS:
            row = {"seed": seed, "lines": lines, "phrasing": phrasing, "key": key, "condition": cond, "expected": truth[key], "skipped": False,
                   "correct": False, "typo_answered": False, "prompt_tokens": None, "wall_ms": None, "lines_for_ms": None, "response": ""}
            t0 = time.perf_counter()
            if cond in agents:
                a = agents[cond]
                if cond == "mem_lines":
                    t1 = time.perf_counter()
                    a.lines_for(key, 3)
                    row["lines_for_ms"] = (time.perf_counter() - t1) * 1000
                text = a.step(q, auto_memory=True, remember_prompt=False, retrieval_query=key).text
            else:
                if cond == "transcript":
                    prompt = transcript + "\n\n" + q
                    if min(int(len(prompt) / CHARS_PER_TOKEN), DECODING.get("num_ctx", 8192)) > cap:
                        row["skipped"] = True
                        rows.append(row)
                        continue
                else:
                    prompt = "RETRIEVED_LINES=\n" + "\n".join(rag_lines(events, key)) + "\n\n" + q
                text = backend.generate(prompt).text
            row["wall_ms"] = (time.perf_counter() - t0) * 1000
            row.update(correct=lenient_correct(text, truth[key]), typo_answered=any(t.lower() in text.lower() for t in typos), response=text,
                       prompt_tokens=getattr(backend.last, "prompt_tokens", None))
            rows.append(row)
    for r in rows:
        r["ingest_s"] = ingest.get(r["condition"])
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for lines in sorted({r["lines"] for r in rows}):
        for ph in sorted({r["phrasing"] for r in rows}):
            for c in CONDITIONS:
                rs = [r for r in rows if (r["lines"], r["phrasing"], r["condition"]) == (lines, ph, c)]
                run = [r for r in rs if not r["skipped"]]
                if not rs:
                    continue
                k = sum(r["correct"] for r in run)
                lf = [r["lines_for_ms"] for r in run if r["lines_for_ms"] is not None]
                ing = [r["ingest_s"] for r in run if r["ingest_s"]]
                out[f"{lines}|{ph}|{c}"] = {
                    "n": len(run), "skipped": len(rs) - len(run), "accuracy": k / len(run) if run else None,
                    "accuracy_ci95": list(wilson(k, len(run))) if run else None,
                    "typo_answered": sum(r["typo_answered"] for r in run) / len(run) if run else None,
                    "prompt_tokens_mean": statistics.fmean(r["prompt_tokens"] for r in run if r["prompt_tokens"]) if any(r["prompt_tokens"] for r in run) else None,
                    "wall_ms_mean": statistics.fmean(r["wall_ms"] for r in run) if run else None,
                    "lines_for_ms_mean": statistics.fmean(lf) if lf else None, "ingest_s_mean": statistics.fmean(ing) if ing else None}
    return out


def run(args) -> dict:
    prepare_model(args.ollama_url, args.model, args.num_ctx)
    inner = OllamaInferenceBackend(model=args.model, base_url=args.ollama_url, timeout_sec=args.timeout) if args.ollama_url \
        else OllamaInferenceBackend(model=args.model, timeout_sec=args.timeout)
    backend = FixedDecodingBackend(inner, num_ctx=args.num_ctx, **DECODING)
    DECODING["num_ctx"] = args.num_ctx
    rows = []
    for lines in args.lines:
        for ph in args.phrasings:
            for seed in args.seeds:
                rows += run_episode(backend, seed, lines, ph, args.transcript_token_cap, args.questions, args.provenance_window)
                print(f"lines={lines} {ph} seed={seed} done", flush=True)
    return {"experiment": "v13_runtime_lines", "version": VERSION, "model": args.model, "seeds": args.seeds, "lines": args.lines, "num_ctx": args.num_ctx,
            "provenance_window": args.provenance_window, "summary": summarize(rows), "rows": rows,
            "boundary": "Synthetic noisy conversations, rule extractor, opaque values, one small model per run."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lines", nargs="+", type=int, default=[100, 1000, 3000])
    p.add_argument("--phrasings", nargs="+", default=["train", "heldout"])
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--questions", type=int, default=6)
    p.add_argument("--transcript-token-cap", type=int, default=7000)
    p.add_argument("--provenance-window", type=int, default=256)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--out", default="results/research-v13-1.5b.json")
    args = p.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'lines':>6} {'phr':<8}{'condition':<13}{'acc':>6}{'typo':>6}{'tokens':>8}{'wall ms':>9}{'lines_for ms':>14}{'ingest s':>10}")
    for k, v in result["summary"].items():
        lines, ph, c = k.split("|")
        acc = "skip" if v["accuracy"] is None else f"{v['accuracy']:.2f}"
        print(f"{lines:>6} {ph:<8}{c:<13}{acc:>6}{(v['typo_answered'] or 0):>6.2f}{(v['prompt_tokens_mean'] or 0):>8.0f}{(v['wall_ms_mean'] or 0):>9.0f}{(v['lines_for_ms_mean'] or 0):>14.3f}{(v['ingest_s_mean'] or 0):>10.2f}")


if __name__ == "__main__":
    main()
