"""NSA Research Benchmark v2: selective persistent memory under long distraction sequences.

The memory conditions use the repository's typed MemoryStore with a deterministic
structured extractor. This intentionally isolates retrieval/storage utility from
LLM extraction quality; extraction is evaluated separately in a future benchmark.
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
from nsa.memory import MemoryItem, MemoryStore
from nsa.runtime.inference.ollama import OllamaInferenceBackend

VALUE_RE = re.compile(r"VALUE\s*=\s*([A-Z0-9_-]+)", re.IGNORECASE)
FACT_RE = re.compile(r"(?:FACT|UPDATE|DISTRACTOR):\s+(ITEM_[0-9A-Z_]+)\s+(?:now\s+)?has value ([A-Z0-9_-]+)", re.IGNORECASE)
CONDITIONS = ("raw", "transcript", "bounded", "nsa_state", "memory_no_cognitive", "nsa_memory")
TASKS = ("recall", "interference", "supersession")


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
    retrieved_expected: bool
    history_limit: int
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
    if task == "supersession":
        new_value = f"V{seed % 1000:03d}_CURRENT"
        observations.append(f"UPDATE: {key} now has value {new_value}; the previous value is obsolete.")
        expected = new_value
    for i in range(distractors):
        k, v = pairs[(i + 1) % len(pairs)]
        observations.append(f"DISTRACTOR: {k} has value {v}.")
    if task == "interference" and distractors >= 2:
        for i in range(distractors // 2):
            k, v = pairs[(i + 9) % len(pairs)]
            observations.append(f"DISTRACTOR: {k} has value {v}.")
    query = f"What is the current value of {key}? Reply with exactly VALUE=<value> and nothing else."
    return observations, query, expected, key


def extract(text: str):
    m = VALUE_RE.search(text or "")
    return m.group(1).upper() if m else None


def target_key(query: str) -> str:
    m = re.search(r"value of (ITEM_[0-9A-Z_]+)", query, re.IGNORECASE)
    if not m:
        raise ValueError(f"could not identify target key from query: {query}")
    return m.group(1).upper()


def build_memory(observations: list[str]) -> MemoryStore:
    store = MemoryStore()
    ordinal = 0
    for observation in observations:
        match = FACT_RE.search(observation)
        if not match:
            continue
        key, value = match.group(1).upper(), match.group(2).upper()
        kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
        store = store.write(
            MemoryItem(
                memory_id=f"m-{ordinal:06d}",
                content={"key": key, "value": value, "kind": kind, "ordinal": ordinal},
                kind=kind,
                provenance_ids=(f"observation-{ordinal}",),
            )
        )
        ordinal += 1
    return store


def retrieve(store: MemoryStore, query: str, limit: int = 3):
    key = target_key(query)
    matches = [
        item for item in store.active()
        if isinstance(item.content, dict) and item.content.get("key") == key
    ]
    return tuple(matches[-limit:])


def memory_context(items) -> str:
    return "\n".join(
        f"- {item.content['key']} = {item.content['value']} ({item.content['kind']}, memory_id={item.memory_id})"
        for item in items
    )


def prompt_for(condition, observations, query, history_limit, retrieved=()):
    if condition == "raw":
        context = ""
    elif condition == "transcript":
        context = "\n".join(observations)
    elif condition in {"bounded", "nsa_state"}:
        context = "\n".join(observations[-history_limit:])
    else:
        context = memory_context(retrieved)
    return (
        "You are a benchmark participant. Use only the supplied information. Do not invent facts.\n"
        + (f"CONTEXT:\n{context}\n" if context else "")
        + query
    )


def run_condition(backend, condition, observations, query, history_limit, memory_limit):
    started = time.perf_counter()
    store = build_memory(observations) if condition.startswith("memory_") else None
    retrieved = retrieve(store, query, memory_limit) if store else ()
    stale = len(retrieved) > 1 and any(i.content.get("kind") == "fact" for i in retrieved[:-1])

    if condition == "nsa_state":
        agent = NSA(
            backend,
            config=RuntimeConfig(
                history_limit=history_limit,
                include_state_in_prompt=True,
                cognitive_enabled=True,
            ),
        )
        for obs in observations:
            agent.observe(obs, source="benchmark")
        result = agent.step(query)
        prompt_chars = len(agent._prompt(query))
        return result.text, prompt_chars, time.perf_counter() - started, 0, 0, False

    if condition == "nsa_memory":
        agent = NSA(
            backend,
            config=RuntimeConfig(
                history_limit=history_limit,
                include_state_in_prompt=False,
                cognitive_enabled=True,
            ),
        )
        memory_prompt = prompt_for(condition, observations, query, history_limit, retrieved)
        for obs in observations:
            agent.observe(obs, source="benchmark")
        result = agent.step(memory_prompt)
        return result.text, len(memory_prompt), time.perf_counter() - started, len(store.items), len(retrieved), stale

    if condition == "memory_no_cognitive":
        prompt = prompt_for(condition, observations, query, history_limit, retrieved)
        result = backend.generate(prompt, max_tokens=32, temperature=0.0)
        return result.text, len(prompt), time.perf_counter() - started, len(store.items), len(retrieved), stale

    prompt = prompt_for(condition, observations, query, history_limit)
    result = backend.generate(prompt, max_tokens=32, temperature=0.0)
    return result.text, len(prompt), time.perf_counter() - started, 0, 0, False


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "raw.jsonl"
    records: list[Record] = []

    with raw_path.open("w", encoding="utf-8") as raw:
        for model in args.models:
            backend = OllamaInferenceBackend(model_name=model, base_url=args.ollama_url)
            for seed in args.seeds:
                for task in TASKS:
                    for distractors in args.distractors:
                        for trial in range(args.trials):
                            episode_seed = seed + trial * 100003
                            observations, query, expected, _ = make_episode(episode_seed, task, distractors)
                            for condition in CONDITIONS:
                                (
                                    text,
                                    prompt_chars,
                                    latency,
                                    memory_items,
                                    retrieved_items,
                                    stale,
                                ) = run_condition(
                                    backend,
                                    condition,
                                    observations,
                                    query,
                                    args.history_limit,
                                    args.memory_limit,
                                )
                                predicted = extract(text)
                                retrieved_expected = False
                                if condition.startswith("memory_"):
                                    store = build_memory(observations)
                                    retrieved = retrieve(store, query, args.memory_limit)
                                    retrieved_expected = any(
                                        i.content.get("value") == expected for i in retrieved
                                    )
                                rec = Record(
                                    model=model,
                                    seed=seed,
                                    condition=condition,
                                    task=task,
                                    distractors=distractors,
                                    trial=trial,
                                    expected=expected,
                                    predicted=predicted,
                                    correct=predicted == expected,
                                    latency_seconds=latency,
                                    prompt_chars=prompt_chars,
                                    response_chars=len(text),
                                    memory_items=memory_items,
                                    retrieved_items=retrieved_items,
                                    retrieval_hit=bool(retrieved_items),
                                    stale_memory_retrieved=stale,
                                    retrieved_expected=retrieved_expected,
                                    history_limit=args.history_limit,
                                    response=text[:2000],
                                )
                                records.append(rec)
                                raw.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                                raw.flush()

    def mean(xs):
        return statistics.fmean(xs) if xs else 0.0

    summary = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.condition == condition]
        summary[condition] = {
            "n": len(rows),
            "accuracy": mean([float(r.correct) for r in rows]),
            "latency_seconds_mean": mean([r.latency_seconds for r in rows]),
            "prompt_chars_mean": mean([r.prompt_chars for r in rows]),
            "response_chars_mean": mean([r.response_chars for r in rows]),
            "retrieval_hit_rate": mean([float(r.retrieval_hit) for r in rows]),
            "retrieved_expected_rate": mean([float(r.retrieved_expected) for r in rows]),
            "stale_memory_retrieval_rate": mean([float(r.stale_memory_retrieved) for r in rows]),
        }

    cells = {}
    for task in TASKS:
        for distractors in args.distractors:
            for condition in CONDITIONS:
                rows = [
                    r for r in records
                    if r.task == task and r.distractors == distractors and r.condition == condition
                ]
                cells[f"{task}/distractors={distractors}/{condition}"] = {
                    "n": len(rows),
                    "accuracy": mean([float(r.correct) for r in rows]),
                    "retrieved_expected_rate": mean([float(r.retrieved_expected) for r in rows]),
                }

    def delta(a, b):
        return {
            "accuracy_delta": summary[a]["accuracy"] - summary[b]["accuracy"],
            "latency_delta_seconds": summary[a]["latency_seconds_mean"] - summary[b]["latency_seconds_mean"],
            "prompt_chars_delta": summary[a]["prompt_chars_mean"] - summary[b]["prompt_chars_mean"],
        }

    result = {
        "benchmark": "NSA Research Benchmark v2",
        "version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "seeds": args.seeds,
        "trials": args.trials,
        "distractors": args.distractors,
        "history_limit": args.history_limit,
        "memory_limit": args.memory_limit,
        "conditions": list(CONDITIONS),
        "hypotheses": {
            "H5": "Selective memory should retain task-relevant facts across long distractor sequences while using substantially less prompt context than a full transcript.",
            "H6": "The retrieval benefit should persist without cognitive-state machinery; nsa_memory should be comparable to memory_no_cognitive, isolating memory retrieval from cognitive-state effects.",
            "H7": "Append-only memory with latest-value retrieval should preserve supersession accuracy without increasing stale-memory retrieval as distractor count grows.",
            "H8": "Any memory accuracy gain must be reported jointly with prompt size, latency, retrieval-hit rate, and stale-memory retrieval rate.",
        },
        "conditions_description": {
            "raw": "Question only; no prior context.",
            "transcript": "Full observation transcript.",
            "bounded": "Only the most recent history_limit observations.",
            "nsa_state": "Current NSARuntime state/cognitive state plus bounded history.",
            "memory_no_cognitive": "Typed MemoryStore + deterministic structured extraction + key retrieval; no cognitive substrate.",
            "nsa_memory": "Typed MemoryStore + same retrieval + NSARuntime cognitive substrate, with memory context explicitly supplied to the model.",
        },
        "extraction_boundary": "Memory extraction is deterministic from the benchmark's FACT/UPDATE schema. This isolates storage/retrieval utility; it does not claim that an LLM can autonomously extract useful memories.",
        "summary": summary,
        "key_deltas": {
            "memory_no_cognitive_vs_bounded": delta("memory_no_cognitive", "bounded"),
            "nsa_memory_vs_memory_no_cognitive": delta("nsa_memory", "memory_no_cognitive"),
            "memory_no_cognitive_vs_transcript": delta("memory_no_cognitive", "transcript"),
        },
        "cells": cells,
        "raw_artifact": str(raw_path),
        "scientific_boundary": "This suite tests selective persistent retrieval in the current runtime. It does not establish consciousness, AGI, general superiority, or autonomous memory formation.",
    }
    (root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    p.add_argument("--distractors", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--history-limit", type=int, default=6)
    p.add_argument("--memory-limit", type=int, default=3)
    p.add_argument("--out", default="results/research-v2")
    p.add_argument("--ollama-url", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
