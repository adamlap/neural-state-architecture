"""NSA Research Benchmark v8: operation under a fixed memory (context) budget.

On a low-memory device the context window is the knob: the KV cache is allocated for ``num_ctx``
tokens when the model loads, so ``num_ctx`` sets resident memory and prefill time. A transcript
agent must keep the whole history inside that window; a retrieval agent only needs the few
lines it retrieves. This benchmark sweeps ``num_ctx`` and the amount of history and records
answer accuracy, whether the prompt was truncated by the context limit, Ollama's own prefill
time, and the resident memory of the model server.

Conditions
* ``transcript``       - the whole history (what a stateless chat client sends). When it does not
                         fit, the server silently drops the *start* of the prompt, which is where
                         the fact is.
* ``rag_lines``        - generic keyword retrieval over the raw log: the last 3 lines that mention
                         the queried key. No structure, no supersession.
* ``temporal_memory``  - the canonical store (versioned, integrity-gated): the current value only.

Transcripts whose processed prompt (``min(estimated tokens, num_ctx)``) exceeds ``--transcript-token-cap`` are
*not run* (recorded as ``skipped``, excluded from accuracy), because a CPU-only box needs minutes per call.

Scientific boundary: the retrieval conditions use a deterministic extractor for the store
(``temporal_memory``) and the question text for keyword retrieval (``rag_lines``); neither tests
LLM-driven memory formation. Memory is measured on the Ollama server (``/api/ps`` size and the
resident set of Ollama processes), so it includes weights; the *difference* across ``num_ctx`` is the KV cache.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.grading import lenient_correct
from experiments.research_common.stats import wilson
from experiments.research_v3.temporal_memory_benchmark import build_temporal_memory, make_episode, target_keys

VERSION = "8.0"
CONDITIONS = ("transcript", "rag_lines", "temporal_memory")
TASKS = ("recall", "supersession_current")
DECODING = {"temperature": 0.0, "max_tokens": 32}
CHARS_PER_TOKEN = 3.3


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def server_rss_bytes() -> int | None:
    """Resident set of all local processes whose command line mentions ollama (Linux only)."""
    total, found = 0, False
    for pid in filter(str.isdigit, os.listdir("/proc")) if os.path.isdir("/proc") else ():
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"ollama" not in f.read():
                    continue
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
                        found = True
        except OSError:
            continue
    return total if found else None


def loaded_model_bytes(base_url: str, model: str) -> int | None:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/ps", timeout=5) as resp:
            for m in json.loads(resp.read().decode()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return int(m.get("size", 0)) or None
    except Exception:
        return None
    return None


def build_prompt(condition: str, observations: list[str], query: str) -> tuple[str, int]:
    """Return ``(prompt, retrieved_item_count)``."""
    key = target_keys(query)[0]
    if condition == "transcript":
        return "\n".join(observations) + "\n\n" + query, len(observations)
    if condition == "rag_lines":
        hits = [o for o in observations if re.search(re.escape(key), o, re.IGNORECASE)][-3:]
        return "RETRIEVED_LINES=\n" + "\n".join(hits) + "\n\n" + query, len(hits)
    store = build_temporal_memory(observations)
    return f"RETRIEVED_MEMORY=\n{store.render([key])}\n\n{query}", len(store.retrieve([key]))


@dataclass
class Record:
    model: str
    num_ctx: int
    condition: str
    task: str
    seed: int
    distractors: int
    trial: int
    expected: str
    skipped: bool
    correct_lenient: bool
    truncated: bool
    timed_out: bool
    prompt_tokens: int | None
    estimated_tokens: int
    retrieved: int
    load_ms: float | None
    prompt_eval_ms: float | None
    eval_ms: float | None
    wall_s: float
    resident_bytes: int | None
    server_rss_bytes: int | None
    response: str


def _load(path: Path) -> list[Record]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Record(**json.loads(line)))
            except Exception:
                continue
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _cell(rows: list[Record]) -> dict:
    run = [r for r in rows if not r.skipped]
    n = len(run)
    out = {"n_run": n, "n_skipped": len(rows) - n}
    if n:
        k = sum(r.correct_lenient for r in run)
        out |= {
            "accuracy_lenient": k / n, "accuracy_ci95": list(wilson(k, n)),
            "truncated_rate": sum(r.truncated for r in run) / n, "timeouts": sum(r.timed_out for r in run),
            "prompt_tokens_mean": _mean(r.prompt_tokens for r in run),
            "prompt_eval_ms_mean": _mean(r.prompt_eval_ms for r in run),
            "wall_s_mean": _mean(r.wall_s for r in run),
            "resident_mib_mean": (lambda v: v / 2**20 if v else None)(_mean(r.resident_bytes for r in run)),
            "server_rss_mib_mean": (lambda v: v / 2**20 if v else None)(_mean(r.server_rss_bytes for r in run)),
        }
    return out


def summarize(records: list[Record]) -> dict:
    out: dict = {"by_model_ctx_condition": {}, "by_model_ctx_condition_history": {}}
    for m in sorted({r.model for r in records}):
        for c in sorted({r.num_ctx for r in records if r.model == m}):
            for cond in CONDITIONS:
                rows = [r for r in records if r.model == m and r.num_ctx == c and r.condition == cond]
                out["by_model_ctx_condition"][f"{m}|{c}|{cond}"] = _cell(rows)
                for d in sorted({r.distractors for r in rows}):
                    out["by_model_ctx_condition_history"][f"{m}|{c}|{cond}|d={d}"] = _cell([r for r in rows if r.distractors == d])
    return out


def write_manifest(root: Path, args, records: list[Record]) -> dict:
    manifest = {
        "benchmark": "NSA Research Benchmark v8 (fixed memory budget)", "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "models": args.models, "seeds": args.seeds,
        "num_ctx": args.num_ctx, "distractors": args.distractors, "decoding": DECODING,
        "transcript_token_cap": args.transcript_token_cap, "conditions": list(CONDITIONS), "tasks": list(TASKS),
        "episodes_recorded": len(records),
        "hypotheses": {
            "H26": "Resident memory of the model server grows with num_ctx (KV cache), so a smaller context budget is a real memory saving.",
            "H27": "Retrieval conditions keep their accuracy at small num_ctx at any history length; the transcript loses it once the history exceeds the budget.",
            "H28": "Prompt-processing time is flat in history length for retrieval and grows with it for the transcript.",
        },
        "scientific_boundary": (
            "Deterministic extractor for temporal_memory; keyword retrieval for rag_lines uses the question text. No LLM memory formation. "
            "Memory is server-side and includes weights; differences across num_ctx are the KV cache. Skipped transcripts are excluded from accuracy."
        ),
        **summarize(records), "raw_artifact": str(root / "raw.jsonl"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw, sidecar = root / "raw.jsonl", root / "config.json"
    signature = {"version": VERSION, "decoding": DECODING, "transcript_token_cap": args.transcript_token_cap}
    if getattr(args, "summarize_only", False):
        return write_manifest(root, args, _load(raw))
    if sidecar.exists() and json.loads(sidecar.read_text()) != signature:
        raise SystemExit(f"{sidecar} was written with a different configuration; use a new --out")
    sidecar.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    records = _load(raw)
    done = {(r.model, r.num_ctx, r.condition, r.task, r.seed, r.distractors, r.trial) for r in records}
    base_url = args.ollama_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    total = len(args.models) * len(args.num_ctx) * len(args.distractors) * len(TASKS) * len(args.seeds) * args.trials * len(CONDITIONS)
    n = len(records)
    with raw.open("a", encoding="utf-8") as handle:
        for model in args.models:
            inner = OllamaInferenceBackend(model=model, base_url=args.ollama_url, timeout_sec=args.timeout) \
                if args.ollama_url else OllamaInferenceBackend(model=model, timeout_sec=args.timeout)
            backend = FixedDecodingBackend(inner, num_ctx=args.num_ctx[0], **DECODING)
            for num_ctx in args.num_ctx:                       # outermost: each change reloads the model
                backend.num_ctx = num_ctx
                for d in args.distractors:
                    for task in TASKS:
                        for seed in args.seeds:
                            for trial in range(args.trials):
                                obs, query, expected, _, _ = make_episode(seed + trial * 100003, task, d, "unique", "plain")
                                for cond in CONDITIONS:
                                    if (model, num_ctx, cond, task, seed, d, trial) in done:
                                        continue
                                    n += 1
                                    prompt, retrieved = build_prompt(cond, obs, query)
                                    est = estimate_tokens(prompt)
                                    skip = cond == "transcript" and min(est, num_ctx) > args.transcript_token_cap   # a truncated prompt costs at most num_ctx
                                    if skip:
                                        rec = Record(model, num_ctx, cond, task, seed, d, trial, expected, True, False, False, False,
                                                     None, est, retrieved, None, None, None, 0.0, None, None, "")
                                    else:
                                        start = time.perf_counter()
                                        out = backend.generate(prompt)
                                        wall = time.perf_counter() - start
                                        g = backend.last
                                        rec = Record(
                                            model, num_ctx, cond, task, seed, d, trial, expected, False,
                                            False if g.timed_out else lenient_correct(out.text, expected),
                                            g.truncated, g.timed_out, g.prompt_tokens, est, retrieved,
                                            g.load_ms, g.prompt_eval_ms, g.eval_ms, round(wall, 3),
                                            loaded_model_bytes(base_url, model), server_rss_bytes(), out.text,
                                        )
                                    records.append(rec)
                                    handle.write(json.dumps(asdict(rec)) + "\n")
                                    handle.flush()
                                    print(f"[{n}/{total}] {model} ctx={num_ctx} d={d} {task} {cond} -> "
                                          f"{'SKIPPED' if rec.skipped else ('OK' if rec.correct_lenient else 'FAIL')}"
                                          f"{' TRUNCATED' if rec.truncated else ''} ({rec.wall_s:.1f}s)", flush=True)
    manifest = write_manifest(root, args, records)
    print(json.dumps({k: {m: v.get(m) for m in ("accuracy_lenient", "resident_mib_mean", "prompt_eval_ms_mean")}
                      for k, v in manifest["by_model_ctx_condition"].items()}, indent=1))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    p.add_argument("--num-ctx", nargs="+", type=int, default=[2048, 4096, 8192, 32768])
    p.add_argument("--distractors", nargs="+", type=int, default=[10, 100, 250, 1000])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--transcript-token-cap", type=int, default=12000)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--out", default="results/research-v8")
    p.add_argument("--ollama-url", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
