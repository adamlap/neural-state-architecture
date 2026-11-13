"""Where does the time of one NSA-shaped model call go, and which request settings cut it?

An NSA call is small: ~130-190 prompt tokens (retrieved memory + question + answer instruction) and a one-value answer. For each variant the script sends the same
questions (unique memory values per call, so nothing but a deliberately shared prefix can hit Ollama's KV prefix cache) and records Ollama's own timings:
prompt tokens actually evaluated, prefill ms, generated tokens, decode ms, wall ms. Variants are cumulative request-level changes; none changes the model weights.

* ``baseline``      - memory first, instruction last, num_ctx 8192, 32-token budget (the layout the benchmarks used)
* ``threads``       - + num_thread = CPU count
* ``small_ctx``     - + num_ctx 2048 (NSA prompts are tiny; a transcript needs the big window, memory does not)
* ``stop``          - + stop at the first newline (the answer is one value; models otherwise keep explaining until the budget)
* ``static_prefix`` - + instruction moved to a constant ``system`` prefix so its tokens are served from the KV prefix cache on every call
* ``tight_budget``  - + 12-token budget

Accuracy is checked too (the value must be in the answer), because a faster wrong answer is not a speedup.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from random import Random

from experiments.research_common.memory_guard import prepare_model

INSTRUCTION = "Answer with the value only."
SYSTEM = "You answer questions using RETRIEVED_MEMORY. " + INSTRUCTION
ENT = ("staging", "billing", "payments", "auth", "search", "reports", "gateway", "mailer")
ATTR = ("database port", "admin email", "api region", "cache size", "queue name", "log level", "owner team", "timeout seconds")
VARIANTS = ("baseline", "threads", "small_ctx", "stop", "static_prefix", "tight_budget")


def case(rng: Random):
    keys = rng.sample([(e, a) for e in ENT for a in ATTR], 3)
    vals = ["".join(rng.choice("bcdfghjklmnpqrstvwxz") + rng.choice("aeiou") for _ in range(3)) + str(rng.randrange(10, 99)) for _ in keys]
    mem = "\n".join(f"- {e} {a} = {v} (trusted)" for (e, a), v in zip(keys, vals))
    e, a = keys[0]
    return mem, f"What is the {e} {a}?", vals[0]


def payload(variant: str, model: str, mem: str, q: str, threads: int) -> dict:
    level = VARIANTS.index(variant)
    opts = {"temperature": 0.0, "num_predict": 32, "num_ctx": 8192}
    body = {"model": model, "stream": False, "options": opts}
    if level >= 1:
        opts["num_thread"] = threads
    if level >= 2:
        opts["num_ctx"] = 2048
    if level >= 3:
        opts["stop"] = ["\n"]
    if level >= 4:
        body["system"] = SYSTEM
        body["prompt"] = f"RETRIEVED_MEMORY=\n{mem}\n\n{q}"
    else:
        body["prompt"] = f"RETRIEVED_MEMORY=\n{mem}\n\n{q} {INSTRUCTION}"
    if level >= 5:
        opts["num_predict"] = 12
    return body


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def run(model: str, url: str, n: int, threads: int, variants) -> dict:
    out = {}
    for variant in variants:
        prepare_model(url, model, 2048 if VARIANTS.index(variant) >= 2 else 8192)
        rng = Random(7)
        post(url, payload(variant, model, *case(Random(99))[:2], threads))                  # load + warm (not timed)
        rows = []
        for _ in range(n):
            mem, q, val = case(rng)
            t0 = time.perf_counter()
            d = post(url, payload(variant, model, mem, q, threads))
            rows.append({"wall_ms": (time.perf_counter() - t0) * 1e3, "prompt_tokens": d.get("prompt_eval_count", 0), "prefill_ms": d.get("prompt_eval_duration", 0) / 1e6,
                         "out_tokens": d.get("eval_count", 0), "decode_ms": d.get("eval_duration", 0) / 1e6, "correct": val.lower() in d.get("response", "").lower(),
                         "response": d.get("response", "")})
        med = lambda k: statistics.median(r[k] for r in rows)
        out[variant] = {"n": n, "wall_ms": med("wall_ms"), "prompt_tokens_evaluated": med("prompt_tokens"), "prefill_ms": med("prefill_ms"), "out_tokens": med("out_tokens"),
                        "decode_ms": med("decode_ms"), "accuracy": sum(r["correct"] for r in rows) / n, "rows": rows}
        v = out[variant]
        print(f"{variant:<14} wall {v['wall_ms']:>7.0f} ms  prefill {v['prefill_ms']:>6.0f} ms ({v['prompt_tokens_evaluated']:.0f} tok)  decode {v['decode_ms']:>6.0f} ms ({v['out_tokens']:.0f} tok)  acc {v['accuracy']:.2f}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    res = {"experiment": "nsa_call_speed", "model": args.model, "threads": args.threads, "summary": run(args.model, args.url, args.n, args.threads, args.variants)}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
