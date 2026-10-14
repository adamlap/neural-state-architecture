"""NSA Research Benchmark v4.1: temporal semantics and retrieval-vs-answer attribution.

v4 keeps the deterministic FACT/UPDATE/DISTRACTOR extractor from v3 but makes
the model-facing temporal semantics explicit and records memory retrieval
correctness separately from final model-answer correctness.

4.1 fixes validity problems in 4.0:

* episodes use a unique-key pool, so the target key can no longer reappear as a
  "DISTRACTOR" carrying its old value (a 64-key pool wrapped around from d>=64);
* every condition is decoded identically (temperature 0, 32 tokens, fixed num_ctx),
  where 4.0 sent NSA conditions at the backend defaults (0.7 / 256 tokens);
* the prompt that was actually sent, token counts, truncation and timeouts are
  recorded instead of being reconstructed afterwards;
* ``nsa_history`` is a matched control (same history window, no state text), since
  ``bounded`` sees raw text while ``nsa_state`` sees a different prompt shape;
* strict and lenient correctness are both reported, plus Wilson CIs, exact McNemar
  on paired episodes and Holm correction;
* stale retrieval is measured from the store's conflict metadata, not hard-coded;
* a manifest is always written (also with ``--summarize-only``) and a sidecar guards
  ``--resume`` against mixing episodes from a different configuration.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nsa.agent import NSA, RuntimeConfig
from nsa.memory import MemoryItem, TemporalMemoryStore
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.episodes import GENERATORS
from experiments.research_common.grading import lenient_correct, strict_correct
from experiments.research_common.stats import holm, mcnemar_exact, wilson
from experiments.research_v3.temporal_memory_benchmark import (
    CONDITIONS as V3_CONDITIONS,
    TASKS,
    temporal_stale,
    FACT_RE,
    VALUE_RE,
    append_retrieve,
    build_append_memory,
    build_temporal_memory,
    make_episode,
    target_keys,
    temporal_retrieve,
)


VERSION = "4.1"
CONDITIONS = ("raw", "transcript", "bounded", "nsa_history", "nsa_state",
              "append_memory", "temporal_memory", "nsa_temporal_memory")
MEMORY_CONDITIONS = ("append_memory", "temporal_memory", "nsa_temporal_memory")
DECODING = {"temperature": 0.0, "max_tokens": 32, "num_ctx": 32768}
assert set(V3_CONDITIONS) <= set(CONDITIONS)


@dataclass
class Record:
    model: str
    seed: int
    condition: str
    task: str
    distractors: int
    trial: int
    expected: str
    predicted: str | None
    correct: bool
    correct_lenient: bool
    memory_retrieval_correct: bool | None
    latency_seconds: float
    prompt_chars: int
    response_chars: int
    memory_items: int
    retrieved_items: int
    retrieval_hit: bool
    stale_memory_retrieved: bool
    history_query: bool
    response: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False
    timed_out: bool = False


def extract(text: str):
    match = VALUE_RE.search(text or "")
    return match.group(1).upper() if match else None


def _item_value(item: MemoryItem) -> str:
    content = item.content
    if isinstance(content, dict):
        return str(content.get("value", "")).upper()
    return str(content).upper()


def _expected_values(expected: str) -> list[str]:
    return [value.upper() for value in expected.split("|")]


def assess_memory_retrieval(
    condition: str,
    observations: list[str],
    query: str,
    expected: str,
    task: str,
    memory_limit: int,
) -> tuple[bool | None, bool]:
    if condition not in MEMORY_CONDITIONS:
        return None, False

    keys = target_keys(query)
    history = task == "supersession_history"
    if condition == "append_memory":
        store = build_append_memory(observations)
        items = append_retrieve(store, keys, history=history, limit=memory_limit)
        stale = task == "supersession_current" and any(
            isinstance(item.content, dict) and item.content.get("kind") == "fact"
            for item in items
        )
    else:
        store = build_temporal_memory(observations)
        items = temporal_retrieve(store, keys, history=history, limit=memory_limit)
        stale = (not history) and temporal_stale(items)

    values = [_item_value(item) for item in items]
    expected_values = _expected_values(expected)
    if task == "multi_current":
        correct = all(value in values for value in expected_values)
    else:
        correct = bool(values) and expected_values[0] in values
    return correct, stale


def render_temporal(store: TemporalMemoryStore, keys: list[str], history: bool, limit: int) -> str:
    return store.render(keys, limit=limit, mode="history" if history else "current")


def _nsa_agent(backend, history_limit: int, **overrides):
    return NSA(backend, config=RuntimeConfig(history_limit=history_limit, cognitive_enabled=True, **overrides))


def answer_for(condition, backend, observations, query, expected, task, history_limit, memory_limit):
    """Return ``(text, prompt_chars, memory_items, retrieved_items, hit, stale)``.

    The prompt length is the length of the prompt the backend actually received
    (``backend.last``), so it can not drift from what the model saw.
    """
    history = task == "supersession_history"
    keys = target_keys(query)

    if condition == "raw":
        out = backend.generate(query)
        return out.text, len(backend.last.prompt), 0, 0, False, False
    if condition == "transcript":
        out = backend.generate("\n".join(observations) + "\n\n" + query)
        return out.text, len(backend.last.prompt), 0, len(observations), True, False
    if condition == "bounded":
        out = backend.generate("\n".join(observations[-history_limit:]) + "\n\n" + query)
        return out.text, len(backend.last.prompt), 0, min(history_limit, len(observations)), True, False
    if condition in ("nsa_history", "nsa_state"):
        overrides = {"include_state_in_prompt": False, "include_history_in_prompt": True} if condition == "nsa_history" else {}
        agent = _nsa_agent(backend, history_limit, **overrides)
        for observation in observations:
            agent.observe(observation)
        result = agent.step(query)
        return result.text, len(backend.last.prompt), 0, 0, True, False

    if condition == "append_memory":
        store = build_append_memory(observations)
        items = append_retrieve(store, keys, history=history, limit=memory_limit)
        stale = task == "supersession_current" and any(
            isinstance(item.content, dict) and item.content.get("kind") == "fact" for item in items
        )
        context = "\n".join(f"- {i.content.get('key')}: {i.content.get('value')}" for i in items) or "NO_RELEVANT_MEMORY"
        out = backend.generate(f"RETRIEVED_MEMORY=\n{context}\n\n{query}")
        return out.text, len(backend.last.prompt), len(store.items), len(items), bool(items), stale

    store = build_temporal_memory(observations)
    items = temporal_retrieve(store, keys, history=history, limit=memory_limit)
    stale = (not history) and temporal_stale(items)
    prompt = f"RETRIEVED_MEMORY=\n{render_temporal(store, keys, history, memory_limit)}\n\n{query}"
    if condition == "temporal_memory":
        out = backend.generate(prompt)
        return out.text, len(backend.last.prompt), len(store.store.items), len(items), bool(items), stale

    agent = _nsa_agent(backend, history_limit, memory_enabled=True, memory_limit=memory_limit)
    for observation in observations:
        match = FACT_RE.search(observation)
        if match:
            key, value = match.group(1).upper(), match.group(2).upper()
            kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
            agent.remember(key, {"value": value, "key": key}, kind=kind)
        agent.observe(observation)
    # History uses the explicit temporal context directly. Current retrieval uses
    # the runtime's memory path so both representations are tested.
    result = agent.step(prompt) if history else agent.step(query, memory_keys=keys)
    return result.text, len(backend.last.prompt), len(store.store.items), len(items), bool(items), stale


def _config_signature(args) -> dict:
    return {
        "version": VERSION, "generator": args.generator, "instruction": args.instruction,
        "history_limit": args.history_limit, "memory_limit": args.memory_limit,
        "decoding": DECODING,
    }


def _check_sidecar(root: Path, signature: dict, resume: bool) -> None:
    """Refuse to resume into a raw.jsonl produced under a different configuration."""
    sidecar = root / "config.json"
    if sidecar.exists() and resume:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing != signature:
            raise SystemExit(
                f"{sidecar} was written with a different configuration ({existing}); "
                "use a new --out or --no-resume"
            )
    sidecar.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")


def _load_records(raw_path: Path) -> list[Record]:
    records: list[Record] = []
    if not raw_path.exists():
        return records
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(Record(**json.loads(line)))
            except Exception:
                continue
    return records


def _rate(rows, attr="correct"):
    return statistics.mean(float(getattr(r, attr)) for r in rows) if rows else None


def _cell(rows) -> dict:
    n = len(rows)
    k = sum(r.correct for r in rows)
    kl = sum(r.correct_lenient for r in rows)
    measured = [r for r in rows if r.memory_retrieval_correct is not None]
    lo, hi = wilson(k, n)
    llo, lhi = wilson(kl, n)
    return {
        "n": n,
        "accuracy": k / n if n else None,
        "accuracy_ci95": [lo, hi],
        "accuracy_lenient": kl / n if n else None,
        "accuracy_lenient_ci95": [llo, lhi],
        "memory_retrieval_correct_rate": _rate(measured, "memory_retrieval_correct"),
        "retrieval_answer_gap": (
            statistics.mean(float(r.memory_retrieval_correct) - float(r.correct_lenient) for r in measured)
            if measured else None
        ),
        "timeouts": sum(r.timed_out for r in rows),
        "truncated": sum(r.truncated for r in rows),
        "latency_seconds_mean": statistics.mean(r.latency_seconds for r in rows) if rows else None,
        "prompt_chars_mean": statistics.mean(r.prompt_chars for r in rows) if rows else None,
        "memory_items_mean": statistics.mean(r.memory_items for r in rows) if rows else None,
        "retrieved_items_mean": statistics.mean(r.retrieved_items for r in rows) if rows else None,
        "retrieval_hit_rate": _rate(rows, "retrieval_hit"),
        "stale_memory_rate": _rate(rows, "stale_memory_retrieved"),
    }


PAIRS = (
    ("nsa_state", "nsa_history"),            # does state text help beyond the same history window?
    ("nsa_history", "bounded"),              # does the NSA wrapper itself change anything?
    ("temporal_memory", "append_memory"),    # canonical vs append-only retrieval
    ("nsa_temporal_memory", "temporal_memory"),
    ("temporal_memory", "transcript"),
    ("append_memory", "transcript"),
)


def paired_tests(records: list[Record], metric: str = "correct_lenient") -> list[dict]:
    outcome = {}
    for r in records:
        outcome.setdefault(r.condition, {})[(r.model, r.seed, r.task, r.distractors, r.trial)] = getattr(r, metric)
    rows = []
    for a, b in PAIRS:
        if a in outcome and b in outcome:
            result = mcnemar_exact(outcome[a], outcome[b])
            rows.append({"first": a, "second": b, "metric": metric, **result})
    for row, adjusted in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adjusted
    return rows


def summarize(records: list[Record]) -> tuple[dict, dict]:
    summary = {c: _cell([r for r in records if r.condition == c]) for c in CONDITIONS if any(r.condition == c for r in records)}
    by_task = {
        t: {c: _cell([r for r in records if r.task == t and r.condition == c])
            for c in CONDITIONS if any(r.task == t and r.condition == c for r in records)}
        for t in TASKS
    }
    return summary, by_task


def write_manifest(root: Path, args, records: list[Record]) -> dict:
    summary, by_task = summarize(records)
    manifest = {
        "benchmark": "NSA Research Benchmark v4",
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "seeds": args.seeds,
        "trials": args.trials,
        "distractors": args.distractors,
        "history_limit": args.history_limit,
        "memory_limit": args.memory_limit,
        "generator": args.generator,
        "instruction": args.instruction,
        "decoding": DECODING,
        "conditions": list(CONDITIONS),
        "tasks": list(TASKS),
        "episodes_recorded": len(records),
        "hypotheses": {
            "H13": "Temporal memory exposes explicit CURRENT versus SUPERSEDED semantics to the model.",
            "H14": "Memory retrieval correctness can be measured independently from final model-answer correctness.",
            "H15": "Historical retrieval remains correct when the previous value is explicitly labeled SUPERSEDED and the current value is provided as CURRENT_VALUE.",
            "H16": "Temporal memory retains its accuracy/latency advantage over cognitive-state-only context without requiring cognitive machinery.",
            "H17": "nsa_state differs from nsa_history (same window, no state text) only if the serialised state carries task-relevant information.",
        },
        "scientific_boundary": (
            "Deterministic extraction isolates storage/retrieval semantics. "
            "v4 does not test autonomous LLM memory formation; the extractor is a keyed lookup oracle."
        ),
        "metrics": "accuracy = strict exact match; accuracy_lenient also accepts wrapper punctuation and echoed VERSION/STATUS suffixes.",
        "summary": summary,
        "by_task": by_task,
        "paired_tests_lenient": paired_tests(records, "correct_lenient"),
        "paired_tests_strict": paired_tests(records, "correct"),
        "raw_artifact": str(root / "raw.jsonl"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "raw.jsonl"
    resume = not getattr(args, "no_resume", False)

    if getattr(args, "summarize_only", False):
        manifest = write_manifest(root, args, _load_records(raw_path))
        print(json.dumps({k: manifest[k] for k in ("version", "episodes_recorded", "paired_tests_lenient")}, indent=2))
        return manifest

    _check_sidecar(root, _config_signature(args), resume)
    if not resume and raw_path.exists():
        raw_path.unlink()
    records = _load_records(raw_path) if resume else []
    completed_keys = {(r.model, r.seed, r.condition, r.task, r.distractors, r.trial) for r in records}
    if completed_keys:
        print(f"Resuming benchmark: {len(completed_keys)} episodes already completed in {raw_path}", flush=True)
    total = (
        len(args.models) * len(args.seeds) * len(args.distractors)
        * args.trials * len(TASKS) * len(CONDITIONS)
    )
    done = len(records)

    with raw_path.open("a", encoding="utf-8") as handle:
        for model in args.models:
            inner = OllamaInferenceBackend(
                model=model, base_url=args.ollama_url, timeout_sec=args.timeout
            ) if args.ollama_url else OllamaInferenceBackend(model=model, timeout_sec=args.timeout)
            backend = FixedDecodingBackend(inner, **DECODING)
            for seed in args.seeds:
                for task in TASKS:
                    for distractors in args.distractors:
                        for trial in range(args.trials):
                            observations, query, expected, _, history_query = make_episode(
                                seed + trial * 100003, task, distractors, args.generator, args.instruction
                            )
                            for condition in CONDITIONS:
                                key = (model, seed, condition, task, distractors, trial)
                                if key in completed_keys:
                                    continue
                                done += 1
                                start = time.perf_counter()
                                response = prompt_chars = None
                                for attempt in range(2):  # one retry on timeout
                                    response, prompt_chars, memory_items, retrieved_items, hit, stale = answer_for(
                                        condition, backend, observations, query, expected,
                                        task, args.history_limit, args.memory_limit,
                                    )
                                    if not backend.last.timed_out:
                                        break
                                    time.sleep(1.0)
                                sent = backend.last
                                text = response.text if hasattr(response, "text") else str(response)
                                retrieval_correct, measured_stale = assess_memory_retrieval(
                                    condition, observations, query, expected, task, args.memory_limit
                                )
                                timed_out = sent.timed_out
                                record = Record(
                                    model=model, seed=seed, condition=condition, task=task,
                                    distractors=distractors, trial=trial, expected=expected,
                                    predicted=extract(text),
                                    correct=False if timed_out else strict_correct(text, expected),
                                    correct_lenient=False if timed_out else lenient_correct(text, expected),
                                    memory_retrieval_correct=retrieval_correct,
                                    latency_seconds=round(time.perf_counter() - start, 4),
                                    prompt_chars=prompt_chars, response_chars=len(text),
                                    memory_items=memory_items, retrieved_items=retrieved_items,
                                    retrieval_hit=hit, stale_memory_retrieved=stale or measured_stale,
                                    history_query=history_query, response=text,
                                    prompt_tokens=sent.prompt_tokens, completion_tokens=sent.completion_tokens,
                                    truncated=sent.truncated, timed_out=timed_out,
                                )
                                records.append(record)
                                handle.write(json.dumps(asdict(record)) + "\n")
                                handle.flush()
                                retrieval_status = (
                                    "n/a" if retrieval_correct is None
                                    else ("MEM_OK" if retrieval_correct else "MEM_FAIL")
                                )
                                note = " (TIMEOUT)" if timed_out else (" (TRUNCATED)" if sent.truncated else "")
                                print(
                                    f"[{done}/{total}] {model} s={seed} {task} d={distractors} {condition} -> "
                                    f"{'OK' if record.correct else ('LENIENT' if record.correct_lenient else 'FAIL')}"
                                    f"{note} {retrieval_status} ({record.latency_seconds:.2f}s)",
                                    flush=True,
                                )

    manifest = write_manifest(root, args, records)
    print(json.dumps({"version": VERSION, "episodes_recorded": len(records),
                      "accuracy_lenient": {c: v["accuracy_lenient"] for c, v in manifest["summary"].items()}}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    parser.add_argument("--distractors", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--memory-limit", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--generator", choices=GENERATORS, default="unique")
    parser.add_argument("--instruction", choices=("legacy", "plain"), default="plain")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild manifest.json from raw.jsonl and exit")
    parser.add_argument("--out", default="results/research-v4.1")
    parser.add_argument("--ollama-url", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
