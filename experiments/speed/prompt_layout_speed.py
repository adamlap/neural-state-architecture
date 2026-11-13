"""Prompt layouts for a memory-backed call: which one makes the model do the least work?

Per-call time on CPU is dominated by prefill (tokens the model must read), see ``nsa_call_speed``. Three layouts over the same 80-fact memory, same questions, all threads:

* ``retrieved3``   - the 3 best-matching memory lines, then the question (what the benchmarks do)
* ``top1``         - only the single best-matching line, then the question
* ``pinned``       - the WHOLE memory as a constant prefix, then the question. The first call pays for the full prefix; afterwards Ollama's KV prefix cache serves it and
                     only the question is evaluated. ``pinned_append`` adds a new fact to the END of the prefix before every 4th question (prefix stays valid);
                     ``pinned_update`` changes a fact in the MIDDLE every 4th question (cache invalid from that line on).

Reports median wall / prefill / decode ms, prompt tokens actually evaluated, and accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from random import Random

from experiments.research_common.memory_guard import prepare_model
from experiments.speed.nsa_call_speed import post

SYSTEM = "You answer questions using RETRIEVED_MEMORY. Answer with the value only."
ENT = ("staging", "billing", "payments", "auth", "search", "reports", "gateway", "mailer", "scheduler", "uploads")
ATTR = ("database port", "admin email", "api region", "cache size", "queue name", "log level", "owner team", "timeout seconds")
LAYOUTS = ("retrieved3", "top1", "pinned", "pinned_append", "pinned_update")


def token(rng: Random) -> str:
    return "".join(rng.choice("bcdfghjklmnpqrstvwxz") + rng.choice("aeiou") for _ in range(3)) + str(rng.randrange(10, 99))


def line(key, val) -> str:
    return f"- {key[0]} {key[1]} = {val}"


def run(model: str, url: str, n: int, threads: int, layouts, num_ctx: int) -> dict:
    out = {}
    for layout in layouts:
        prepare_model(url, model, num_ctx)
        rng = Random(11)
        keys = [(e, a) for e in ENT for a in ATTR]
        rng.shuffle(keys)
        mem = {k: token(rng) for k in keys}
        extra = 0
        rows = []
        for i in range(n + 1):                                             # call 0 warms / pays the pinned prefix; not scored
            if i and i % 4 == 0 and layout == "pinned_append":
                extra += 1
                mem[(f"service{extra}", "api region")] = token(rng)
            if i and i % 4 == 0 and layout == "pinned_update":
                mem[list(mem)[len(mem) // 2]] = token(rng)
            key = rng.choice(keys)
            q = f"What is the {key[0]} {key[1]}?"
            if layout.startswith("pinned"):
                ctx = "\n".join(line(k, v) for k, v in mem.items())
            else:
                # the same overlap ranking the runtime uses: best key first, then the nearest distractors (same entity / same attribute)
                near = [key] + [k for k in mem if k != key and (k[0] == key[0] or k[1] == key[1])][: 0 if layout == "top1" else 2]
                ctx = "\n".join(line(k, mem[k]) for k in near)
            body = {"model": model, "stream": False, "system": SYSTEM, "prompt": f"RETRIEVED_MEMORY=\n{ctx}\n\n{q}",
                    "options": {"temperature": 0.0, "num_predict": 16, "num_ctx": num_ctx, "num_thread": threads, "stop": ["\n"]}}
            t0 = time.perf_counter()
            d = post(url, body)
            if i:
                rows.append({"wall_ms": (time.perf_counter() - t0) * 1e3, "prompt_tokens": d.get("prompt_eval_count", 0), "prefill_ms": d.get("prompt_eval_duration", 0) / 1e6,
                             "out_tokens": d.get("eval_count", 0), "decode_ms": d.get("eval_duration", 0) / 1e6, "correct": mem[key].lower() in d.get("response", "").lower(),
                             "response": d.get("response", "")})
            else:
                first = (time.perf_counter() - t0) * 1e3
        med = lambda k: statistics.median(r[k] for r in rows)
        out[layout] = {"n": n, "first_call_ms": first, "wall_ms": med("wall_ms"), "wall_ms_mean": statistics.fmean(r["wall_ms"] for r in rows), "prompt_tokens_evaluated": med("prompt_tokens"),
                       "prefill_ms": med("prefill_ms"), "decode_ms": med("decode_ms"), "accuracy": sum(r["correct"] for r in rows) / n, "rows": rows}
        v = out[layout]
        print(f"{layout:<14} first {first:>7.0f}  wall med {v['wall_ms']:>6.0f} mean {v['wall_ms_mean']:>6.0f} ms  prefill {v['prefill_ms']:>6.0f} ms ({v['prompt_tokens_evaluated']:.0f} tok)  "
              f"decode {v['decode_ms']:>5.0f} ms  acc {v['accuracy']:.2f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = {"experiment": "prompt_layout_speed", "model": args.model, "threads": args.threads, "num_ctx": args.num_ctx,
           "summary": run(args.model, args.url, args.n, args.threads, args.layouts, args.num_ctx)}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
