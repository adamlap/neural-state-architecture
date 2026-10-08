"""NSA Research Benchmark v3: temporal canonical memory under long distraction sequences.

v3 extends the v2 selective-memory benchmark with version chains, canonical current
value retrieval, and explicit historical retrieval. The extractor remains
deterministic so the experiment isolates temporal storage/retrieval semantics.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from nsa.agent import NSA, RuntimeConfig
from nsa.memory import MemoryItem, MemoryStore, TemporalMemoryStore
from nsa.runtime.inference.ollama import OllamaInferenceBackend

VALUE_RE = re.compile(r"VALUE\s*=\s*([A-Z0-9_-]+)", re.IGNORECASE)
FACT_RE = re.compile(r"(?:FACT|UPDATE|DISTRACTOR):\s+(ITEM_[0-9A-Z_]+)\s+(?:now\s+)?has value ([A-Z0-9_-]+)", re.IGNORECASE)
CONDITIONS = ("raw", "transcript", "bounded", "nsa_state", "append_memory", "temporal_memory", "nsa_temporal_memory")
TASKS = ("recall", "interference", "supersession_current", "supersession_history", "multi_current")


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
    latency_seconds: float
    prompt_chars: int
    response_chars: int
    memory_items: int
    retrieved_items: int
    retrieval_hit: bool
    stale_memory_retrieved: bool
    history_query: bool
    response: str


def facts(seed: int, n: int = 64):
    rng = Random(seed)
    keys = [f"ITEM_{i:02d}" for i in range(n)]
    rng.shuffle(keys)
    return [(k, f"V{seed % 1000:03d}_{i:03d}") for i, k in enumerate(keys)]


def make_episode(seed: int, task: str, distractors: int):
    pairs = facts(seed)
    key, old_value = pairs[0]
    observations = [f"FACT: {key} has value {old_value}."]
    expected = old_value
    query = f"What is the current value of {key}? Reply with exactly VALUE=<value> and nothing else."
    history_query = False
    if task.startswith("supersession"):
        new_value = f"V{seed % 1000:03d}_CURRENT"
        observations.append(f"UPDATE: {key} now has value {new_value}; the previous value is obsolete.")
        expected = new_value
        if task == "supersession_history":
            expected = old_value
            history_query = True
            query = f"What was the previous value of {key}? Reply with exactly VALUE=<value> and nothing else."
    for i in range(distractors):
        k, v = pairs[(i + 1) % len(pairs)]
        observations.append(f"DISTRACTOR: {k} has value {v}.")
    if task == "interference" and distractors >= 2:
        for i in range(distractors // 2):
            k, v = pairs[(i + 9) % len(pairs)]
            observations.append(f"DISTRACTOR: {k} has value {v}.")
    if task == "multi_current":
        selected = pairs[:3]
        observations = [f"FACT: {k} has value {v}." for k, v in selected] + observations
        keys = [k for k, _ in selected]
        expected = "|".join(v for _, v in selected)
        query = f"What are the current values of {', '.join(keys)}? Reply with exactly VALUE=<value1>|<value2>|<value3> and nothing else."
    return observations, query, expected, key, history_query


def extract(text: str):
    m = VALUE_RE.search(text or "")
    return m.group(1).upper() if m else None


def target_keys(query: str) -> list[str]:
    return [x.upper() for x in re.findall(r"ITEM_[0-9A-Z_]+", query, re.IGNORECASE)]


def build_append_memory(observations: list[str]) -> MemoryStore:
    store = MemoryStore()
    for ordinal, observation in enumerate(observations):
        match = FACT_RE.search(observation)
        if not match:
            continue
        key, value = match.group(1).upper(), match.group(2).upper()
        kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
        store = store.write(MemoryItem(
            memory_id=f"m-{ordinal:06d}",
            content={"key": key, "value": value, "kind": kind, "ordinal": ordinal},
            kind=kind,
            provenance_ids=(f"observation-{ordinal}",),
        ))
    return store


def build_temporal_memory(observations: list[str]) -> TemporalMemoryStore:
    store = TemporalMemoryStore()
    for ordinal, observation in enumerate(observations):
        match = FACT_RE.search(observation)
        if not match:
            continue
        key, value = match.group(1).upper(), match.group(2).upper()
        kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
        store = store.remember(
            key, {"value": value, "key": key, "ordinal": ordinal},
            kind=kind, provenance_ids=(f"observation-{ordinal}",),
        )
    return store


def append_retrieve(store: MemoryStore, keys: list[str], *, history: bool, limit: int):
    matched = [
        item for item in store.active()
        if isinstance(item.content, dict) and item.content.get("key", "").upper() in keys
    ]
    matched.sort(key=lambda x: x.content.get("ordinal", 0))
    if history:
        return tuple(matched[-min(limit, len(matched)):])
    # Deliberately preserve recent historical versions: this is the v2-style
    # append-only baseline that exposes the supersession burden to the LLM.
    return tuple(matched[-limit:])


def temporal_retrieve(store: TemporalMemoryStore, keys: list[str], *, history: bool, limit: int):
    if history:
        items = []
        for key in keys:
            versions = store.history(key, limit=2)
            if len(versions) >= 2:
                items.append(versions[1])
            elif versions:
                items.append(versions[0])
        return tuple(items[:limit])
    return store.retrieve(keys, limit=limit)


def render(items) -> str:
    if not items:
        return "NO_RELEVANT_MEMORY"
    rows = []
    for item in items:
        c = item.content
        key = c.get("key") or c.get("memory_key")
        value = c.get("value", c)
        version = c.get("version")
        suffix = f" (version {version})" if version else ""
        rows.append(f"- {key}: {value}{suffix}")
    return "\n".join(rows)


def answer_for(condition: str, backend, observations, query, expected, task, history_limit, memory_limit):
    if condition == "raw":
        prompt = query
        return backend.generate(prompt), len(prompt), 0, 0, False, False
    if condition == "transcript":
        prompt = "\n".join(observations) + "\n\n" + query
        return backend.generate(prompt), len(prompt), 0, len(observations), True, False
    if condition == "bounded":
        prompt = "\n".join(observations[-history_limit:]) + "\n\n" + query
        return backend.generate(prompt), len(prompt), 0, min(history_limit, len(observations)), True, False
    if condition == "nsa_state":
        agent = NSA(backend, config=RuntimeConfig(history_limit=history_limit, cognitive_enabled=True))
        for observation in observations:
            agent.observe(observation)
        result = agent.step(query)
        return result.text, len(agent._prompt(query)), 0, 0, True, False

    keys = target_keys(query)
    history = task == "supersession_history"
    if condition == "append_memory":
        store = build_append_memory(observations)
        items = append_retrieve(store, keys, history=history, limit=memory_limit)
        stale = task == "supersession_current" and any(
            item.content.get("kind") == "fact" for item in items
        )
        context = render(items)
        prompt = f"RETRIEVED_MEMORY=\n{context}\n\n{query}"
        return backend.generate(prompt), len(prompt), len(store.items), len(items), bool(items), stale

    store = build_temporal_memory(observations)
    items = temporal_retrieve(store, keys, history=history, limit=memory_limit)
    stale = False
    context = render(items)
    prompt = f"RETRIEVED_MEMORY=\n{context}\n\n{query}"
    if condition == "temporal_memory":
        raw = backend.generate(prompt)
        return raw, len(prompt), len(store.store.items), len(items), bool(items), stale

    agent = NSA(backend, config=RuntimeConfig(
        history_limit=history_limit, cognitive_enabled=True, memory_enabled=True, memory_limit=memory_limit
    ))
    for observation in observations:
        match = FACT_RE.search(observation)
        if match:
            key, value = match.group(1).upper(), match.group(2).upper()
            kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
            agent.remember(key, {"value": value, "key": key}, kind=kind)
        agent.observe(observation)
    if history:
        result = agent.step(prompt)
    else:
        result = agent.step(query, memory_keys=keys)
    model_prompt = agent._prompt(query if not history else prompt)
    return result.text, len(model_prompt), len(store.store.items), len(items), bool(items), stale


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "raw.jsonl"
    records = []
    total = len(args.models) * len(args.seeds) * len(args.distractors) * args.trials * len(TASKS) * len(CONDITIONS)
    done = 0
    with raw_path.open("a", encoding="utf-8") as handle:
        for model in args.models:
            backend = OllamaInferenceBackend(model=model, base_url=args.ollama_url) if args.ollama_url else OllamaInferenceBackend(model=model)
            for seed in args.seeds:
                for task in TASKS:
                    for distractors in args.distractors:
                        for trial in range(args.trials):
                            observations, query, expected, _, history_query = make_episode(seed + trial, task, distractors)
                            for condition in CONDITIONS:
                                done += 1
                                start = time.perf_counter()
                                response, prompt_chars, memory_items, retrieved_items, hit, stale = answer_for(
                                    condition, backend, observations, query, expected, task, args.history_limit, args.memory_limit
                                )
                                text = response.text if hasattr(response, "text") else str(response)
                                predicted = extract(text)
                                correct = predicted == expected if "|" not in expected else text.strip().upper() == f"VALUE={expected}"
                                record = Record(
                                    model=model, seed=seed, condition=condition, task=task,
                                    distractors=distractors, trial=trial, expected=expected,
                                    predicted=predicted, correct=correct,
                                    latency_seconds=round(time.perf_counter() - start, 4),
                                    prompt_chars=prompt_chars, response_chars=len(text),
                                    memory_items=memory_items, retrieved_items=retrieved_items,
                                    retrieval_hit=hit, stale_memory_retrieved=stale,
                                    history_query=history_query, response=text,
                                )
                                records.append(record)
                                handle.write(json.dumps(asdict(record)) + "\n")
                                handle.flush()
                                print(f"[{done}/{total}] {model} s={seed} {task} d={distractors} {condition} -> {'OK' if correct else 'FAIL'} ({record.latency_seconds:.2f}s)", flush=True)
    summary = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.condition == condition]
        summary[condition] = {
            "n": len(rows),
            "accuracy": statistics.mean(float(r.correct) for r in rows),
            "latency_seconds_mean": statistics.mean(r.latency_seconds for r in rows),
            "prompt_chars_mean": statistics.mean(r.prompt_chars for r in rows),
            "memory_items_mean": statistics.mean(r.memory_items for r in rows),
            "retrieved_items_mean": statistics.mean(r.retrieved_items for r in rows),
            "retrieval_hit_rate": statistics.mean(float(r.retrieval_hit) for r in rows),
            "stale_memory_rate": statistics.mean(float(r.stale_memory_retrieved) for r in rows),
        }
    by_task = {}
    for task in TASKS:
        by_task[task] = {}
        for condition in CONDITIONS:
            rows = [r for r in records if r.task == task and r.condition == condition]
            by_task[task][condition] = {"n": len(rows), "accuracy": statistics.mean(float(r.correct) for r in rows)}
    manifest = {
        "benchmark": "NSA Research Benchmark v3",
        "version": "3.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models, "seeds": args.seeds, "trials": args.trials,
        "distractors": args.distractors, "history_limit": args.history_limit, "memory_limit": args.memory_limit,
        "conditions": list(CONDITIONS), "tasks": list(TASKS),
        "hypotheses": {
            "H9": "Temporal canonical memory should preserve current-value accuracy across long distractor sequences while keeping retrieved context bounded.",
            "H10": "Canonical retrieval should eliminate stale-memory retrieval for current-value queries compared with append-only retrieval.",
            "H11": "Explicit historical retrieval should recover superseded values without contaminating current-value retrieval.",
            "H12": "Temporal memory should provide these gains without requiring cognitive-state machinery; compare temporal_memory with nsa_temporal_memory.",
        },
        "conditions_description": {
            "raw": "Question only.",
            "transcript": "Full observation transcript.",
            "bounded": "Recent observations only.",
            "nsa_state": "Current NSARuntime state/cognitive state plus bounded history.",
            "append_memory": "v2 append-only MemoryStore retrieval.",
            "temporal_memory": "Versioned TemporalMemoryStore with canonical current-value retrieval.",
            "nsa_temporal_memory": "TemporalMemoryStore retrieval plus NSARuntime cognitive substrate.",
        },
        "scientific_boundary": "Deterministic extraction isolates temporal storage/retrieval semantics. This benchmark does not test autonomous LLM memory extraction.",
        "summary": summary, "by_task": by_task, "raw_artifact": str(raw_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    p.add_argument("--distractors", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--history-limit", type=int, default=6)
    p.add_argument("--memory-limit", type=int, default=3)
    p.add_argument("--out", default="results/research-v3")
    p.add_argument("--ollama-url", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
