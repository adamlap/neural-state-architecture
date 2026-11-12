"""Measure prefill and decode speed against ANY Ollama server (local, WSL, or a native Windows host).

Run the same command against two URLs to compare platforms, e.g. WSL:  --ollama-url http://localhost:11434   and Windows:  --ollama-url http://<host>:11434.
For each model and prompt length it reports the median prefill speed (tokens/s from Ollama's own ``prompt_eval_duration``, prompts with a unique prefix so the prefix cache is
not measured), the median decode speed, and the loaded memory from ``/api/ps`` (server-side, so it is valid for a remote host). The transcript-vs-retrieval comparison is the
ratio of prefill time at the long length to the short length. Nothing here starts, stops or reconfigures a server.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .ollama_flags import get, measure


def run(args) -> dict:
    base = args.ollama_url.rstrip("/")
    version = get(f"{base}/api/version").get("version") if args.ollama_url else None
    rows = []
    for model in args.models:
        for tokens in args.tokens:
            m = measure(base, model, tokens, args.num_thread, args.repeats)
            rows.append({"model": model, "tokens": tokens, "num_thread": args.num_thread or "default", **m,
                         "prefill_seconds": m["prompt_tokens"] / m["prefill_tok_s"]})
            print(f"{model:<16}{tokens:>6} tok  prefill {m['prefill_tok_s']:8.1f} tok/s ({rows[-1]['prefill_seconds']:6.1f} s)  decode {m['decode_tok_s']:6.2f} tok/s  "
                  f"{m['resident_mib'] or 0:7.0f} MiB", flush=True)
    return {"experiment": "remote_speed_v1", "ollama_url": base, "ollama_version": version, "label": args.label, "repeats": args.repeats,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows,
            "boundary": "Synthetic prompts; median of a few runs; throughput of one request at a time."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "qwen2.5:3b"])
    p.add_argument("--tokens", nargs="+", type=int, default=[120, 1000, 4000])
    p.add_argument("--num-thread", type=int, default=None)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--label", default="unlabelled")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = run(args)
    out = Path(args.out or f"results/speed_remote_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
