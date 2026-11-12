"""NSA Research Benchmark v11: long histories AND noise together.

v8 showed that retrieval beats a transcript on very long histories (cost and truncation); v10 showed that routing keeps a noisy memory correct. Neither
tested both. Here the v10 timeline (facts, typo'd and clean reminders, genuine and one-character updates, untrusted forwarded updates, chatter) is scaled
to 100 / 300 / 1000 lines over all 80 (entity, attribute) keys and a model answers "what is the current value of <key>?" for sampled keys.

Conditions (same model, decoding and prompts as v10 end to end): ``transcript`` (the whole conversation; silently truncated from the front when it does not fit
``num_ctx``), ``rag_overlap`` (3 most similar lines), ``nsa_rules`` (memory built line by line with the rule extractor, no routing) and ``nsa_rules_routed``.
The ``train`` phrasing is used so that the rule extractor can ingest 1000 lines without a model call; extraction coverage on unfamiliar phrasing is v10's topic.

Scores: accuracy, whether the answer repeats a typo, prompt tokens and Ollama's own prompt-evaluation time, truncation. Transcripts whose processed prompt
(min(estimated tokens, num_ctx)) exceeds ``--transcript-token-cap`` are skipped and excluded.
"""
from __future__ import annotations

import argparse
import json
import statistics
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

VERSION = "11.0"
CONDITIONS = ("transcript", "rag_overlap", "nsa_rules", "nsa_rules_routed")
DECODING = {"temperature": 0.0, "max_tokens": 32}
CHARS_PER_TOKEN = 3.3


def long_stream(seed: int, lines: int):
    """A timeline of about ``lines`` lines over all 80 keys (80 facts, then reminders/updates/chatter in fixed proportions)."""
    body = max(0, lines - 80)
    return make_stream(seed, "train", n_facts=80, reminders=int(body * 0.735), updates=max(2, body // 20), poisoned=max(2, body // 25),
                       near_updates=max(2, body // 40), chatter=int(body * 0.15))


def build_agent(events, route: bool) -> NSA:
    class _Null:
        model = "null"

        def generate(self, prompt, **kw):
            raise RuntimeError("no model for rule ingest")

    a = NSA(_Null(), config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False,
                                          untrusted_sources=(UNTRUSTED_SOURCE,), route_noisy_updates=route), extractor=RuleFactExtractor())
    for text, source in events:
        a.observe(text, source=source)
    return a


def run_episode(backend, seed: int, lines: int, num_ctx: int, cap: int, n_questions: int) -> list[dict]:
    events, truth, typos = long_stream(seed, lines)
    keys = Random(seed).sample(sorted(truth), n_questions)
    agents = {"nsa_rules": build_agent(events, False), "nsa_rules_routed": build_agent(events, True)}
    transcript = "\n".join(t for t, _ in events)
    rows = []
    for key in keys:
        q = f"What is the {key}? {INSTRUCTION}"
        for cond in CONDITIONS:
            skipped = False
            if cond == "transcript":
                prompt = transcript + "\n\n" + q
                skipped = min(int(len(prompt) / CHARS_PER_TOKEN), num_ctx) > cap
            elif cond == "rag_overlap":
                prompt = "RETRIEVED_LINES=\n" + "\n".join(rag_lines(events, key)) + "\n\n" + q
            else:
                a = agents[cond]
                found = a.memory_keys_for(key)
                prompt = f"RETRIEVED_MEMORY=\n{a.memory.render(found) if found else 'NO_RELEVANT_MEMORY'}\n\n{q}"
            row = {"seed": seed, "lines": lines, "num_ctx": num_ctx, "key": key, "condition": cond, "expected": truth[key], "skipped": skipped,
                   "correct": False, "typo_answered": False, "truncated": False, "silently_truncated": False, "prompt_chars": len(prompt),
                   "prompt_tokens": None, "prompt_eval_ms": None, "response": ""}
            if not skipped:
                out = backend.generate(prompt)
                g = backend.last
                # Ollama can drop most of an over-long prompt without reporting a full context (observed: ~4.1k tokens evaluated for a ~14k-token prompt at
                # num_ctx 8192), so also compare the evaluated token count with a character-based estimate.
                silent = bool(g.prompt_tokens) and g.prompt_tokens < 0.6 * (len(prompt) / CHARS_PER_TOKEN)
                row.update(correct=lenient_correct(out.text, truth[key]), typo_answered=any(t.lower() in out.text.lower() for t in typos),
                           truncated=g.truncated, silently_truncated=silent, prompt_tokens=g.prompt_tokens, prompt_eval_ms=g.prompt_eval_ms, response=out.text)
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for lines in sorted({r["lines"] for r in rows}):
        for ctx in sorted({r["num_ctx"] for r in rows if r["lines"] == lines}):
            for c in CONDITIONS:
                rs = [r for r in rows if (r["lines"], r["num_ctx"], r["condition"]) == (lines, ctx, c)]
                run = [r for r in rs if not r["skipped"]]
                if not rs:
                    continue
                k = sum(r["correct"] for r in run)
                out[f"{lines}|{ctx}|{c}"] = {
                    "n": len(run), "skipped": len(rs) - len(run), "accuracy": k / len(run) if run else None,
                    "accuracy_ci95": list(wilson(k, len(run))) if run else None,
                    "typo_answered": sum(r["typo_answered"] for r in run) / len(run) if run else None,
                    "truncated_rate": sum(r["truncated"] or r.get("silently_truncated", False) for r in run) / len(run) if run else None,
                    "prompt_tokens_mean": statistics.fmean(r["prompt_tokens"] for r in run) if run else None,
                    "prompt_eval_ms_mean": statistics.fmean(r["prompt_eval_ms"] for r in run) if run else None}
    return out


def run(args) -> dict:
    prepare_model(args.ollama_url, args.model, max(args.num_ctx))
    inner = OllamaInferenceBackend(model=args.model, base_url=args.ollama_url, timeout_sec=args.timeout) if args.ollama_url \
        else OllamaInferenceBackend(model=args.model, timeout_sec=args.timeout)
    backend = FixedDecodingBackend(inner, num_ctx=args.num_ctx[0], **DECODING)
    rows = []
    for ctx in args.num_ctx:
        backend.num_ctx = ctx
        for lines in args.lines:
            for seed in args.seeds:
                rows += run_episode(backend, seed, lines, ctx, args.transcript_token_cap, args.questions)
                print(f"ctx={ctx} lines={lines} seed={seed} done", flush=True)
    return {"experiment": "v11_long_noisy", "version": VERSION, "model": args.model, "seeds": args.seeds, "lines": args.lines, "num_ctx": args.num_ctx,
            "transcript_token_cap": args.transcript_token_cap, "decoding": DECODING, "summary": summarize(rows), "rows": rows,
            "boundary": "Synthetic noisy conversations, train phrasing (rule extraction), opaque values, one small model."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lines", nargs="+", type=int, default=[100, 300, 1000])
    p.add_argument("--num-ctx", nargs="+", type=int, default=[8192, 32768])
    p.add_argument("--questions", type=int, default=6)
    p.add_argument("--transcript-token-cap", type=int, default=20000)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--out", default="results/research-v11-1.5b.json")
    args = p.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'lines':>6}{'ctx':>7} {'condition':<18}{'acc':>6}{'typo':>6}{'trunc':>7}{'tokens':>8}{'prefill ms':>12}")
    for k, v in result["summary"].items():
        lines, ctx, c = k.split("|")
        acc = "skip" if v["accuracy"] is None else f"{v['accuracy']:.2f}"
        print(f"{lines:>6}{ctx:>7} {c:<18}{acc:>6}{(v['typo_answered'] or 0):>6.2f}{(v['truncated_rate'] or 0):>7.2f}{(v['prompt_tokens_mean'] or 0):>8.0f}{(v['prompt_eval_ms_mean'] or 0):>12.0f}")


if __name__ == "__main__":
    main()
