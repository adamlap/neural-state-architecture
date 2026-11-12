"""Which Ollama serving settings speed up prompt processing and decoding on this machine?

Restarts a private Ollama server for each environment configuration, then for each request-level thread setting measures, per model and prompt length,
median prompt-processing speed (tokens/s of Ollama's own ``prompt_eval_duration``), decode speed (``eval_duration``) and the loaded memory (``/api/ps``).
Server settings tested: default, ``OLLAMA_FLASH_ATTENTION=1``, flash attention + ``OLLAMA_KV_CACHE_TYPE=q8_0`` (quantised KV cache). Request settings:
default threads vs ``num_thread`` = number of CPU cores. Only a server this script starts is touched; an existing Ollama server on the same port is stopped first
(by PID) and NOT restarted, so restart yours afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

OLLAMA = os.path.expanduser("~/.local/ollama/bin/ollama")
CONFIGS = {"default": {}, "flash_attention": {"OLLAMA_FLASH_ATTENTION": "1"},
           "flash_attention+kv_q8": {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}}
WORDS = ("the staging database port is set to a value that operations changed last quarter while billing kept its admin email and the search team "
         "rotated the api region for reports gateway mailer scheduler uploads ").split()


def prompt_of(tokens: int, salt: int = 0) -> str:
    """A prompt whose FIRST tokens are unique per call: Ollama reuses the KV cache for a shared prefix, so a unique suffix would measure the cache, not prefill."""
    n = int(tokens * 0.75)
    return f"Request {salt}-{salt * 7919 % 10007}. " + " ".join(WORDS[(i + salt) % len(WORDS)] + (str(i) if i % 17 == 0 else "") for i in range(n)) + "\n\nSummarise in one word."


def post(url: str, payload: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stop_servers() -> None:
    me = os.getpid()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if int(pid) != me and len(cmd) > 1 and cmd[0].endswith(b"ollama") and cmd[1] == b"serve":
            os.kill(int(pid), signal.SIGTERM)
    time.sleep(3)


def start_server(env: dict, base: str) -> subprocess.Popen:
    e = {**os.environ, "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_NUM_PARALLEL": "1", **env}
    proc = subprocess.Popen([OLLAMA, "serve"], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=os.path.expanduser("~"))
    for _ in range(60):
        try:
            get(f"{base}/api/tags")
            return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("ollama did not start")


def measure(base: str, model: str, tokens: int, threads: int | None, repeats: int) -> dict:
    opts = {"temperature": 0.0, "num_predict": 24, "num_ctx": 4096}
    if threads:
        opts["num_thread"] = threads
    post(f"{base}/api/generate", {"model": model, "prompt": "warm up", "stream": False, "options": opts})   # load + warm
    runs = []
    for i in range(repeats):
        r = post(f"{base}/api/generate", {"model": model, "prompt": prompt_of(tokens, i + 1), "stream": False, "options": opts})   # unique prefix defeats the cache
        runs.append(r)
    expected = runs[0]["prompt_eval_count"]
    # a run that evaluated far fewer tokens than the first was served (mostly) from the prefix cache and says nothing about prefill speed
    pe = [r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9) for r in runs if r.get("prompt_eval_duration") and r["prompt_eval_count"] >= 0.8 * expected]
    ev = [r["eval_count"] / (r["eval_duration"] / 1e9) for r in runs if r.get("eval_duration")]
    mem = next((m["size"] for m in get(f"{base}/api/ps").get("models", []) if m["name"] == model), None)
    return {"prefill_tok_s": statistics.median(pe), "decode_tok_s": statistics.median(ev), "prompt_tokens": runs[-1]["prompt_eval_count"],
            "resident_mib": mem / 2**20 if mem else None}


def run(args) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    cores = os.cpu_count() or 4
    rows = []
    stop_servers()
    for name, env in CONFIGS.items():
        proc = start_server({**env, "OLLAMA_HOST": f"127.0.0.1:{args.port}"}, base)
        try:
            for model in args.models:
                for tokens in args.tokens:
                    for threads in (None, cores):
                        m = measure(base, model, tokens, threads, args.repeats)
                        rows.append({"config": name, "model": model, "tokens": tokens, "threads": threads or "default", **m})
                        print(f"{name:<22}{model:<14}{tokens:>6} tok threads={str(threads or 'default'):<8} prefill {m['prefill_tok_s']:7.1f} tok/s  "
                              f"decode {m['decode_tok_s']:6.2f} tok/s  {m['resident_mib'] or 0:7.0f} MiB", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            time.sleep(2)
    return {"experiment": "ollama_flags_v1", "cores": cores, "repeats": args.repeats, "rows": rows,
            "boundary": "One CPU-only WSL machine; median of a few runs; prompts are synthetic text; the existing Ollama server was stopped and is not restarted."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b", "qwen2.5:3b"])
    p.add_argument("--tokens", nargs="+", type=int, default=[120, 1000])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--out", default="results/speed_ollama_flags_v1.json")
    args = p.parse_args()
    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
