"""NSA Research Benchmark v4: temporal semantics and retrieval-vs-answer attribution.

v4 keeps the deterministic FACT/UPDATE/DISTRACTOR extractor from v3 but makes
the model-facing temporal semantics explicit and records memory retrieval
correctness separately from final model-answer correctness.
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
from nsa.memory import MemoryItem, MemoryStore, TemporalMemoryStore
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_v3.temporal_memory_benchmark import (
    CONDITIONS,
    TASKS,
    FACT_RE,
    VALUE_RE,
    append_retrieve,
    build_append_memory,
    build_temporal_memory,
    make_episode,
    target_keys,
    temporal_retrieve,
)


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
    if condition not in {"append_memory", "temporal_memory", "nsa_temporal_memory"}:
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
        stale = False

    values = [_item_value(item) for item in items]
    expected_values = _expected_values(expected)
    if task == "multi_current":
        correct = all(value in values for value in expected_values)
    else:
        correct = bool(values) and expected_values[0] in values
    return correct, stale


def render_temporal(store: TemporalMemoryStore, keys: list[str], history: bool, limit: int) -> str:
    return store.render(keys, limit=limit, mode="history" if history else "current")


def answer_for(
    condition: str,
    backend,
    observations: list[str],
    query: str,
    expected: str,
    task: str,
    history_limit: int,
    memory_limit: int,
):
    history = task == "supersession_history"

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
        agent = NSA(backend, config=RuntimeConfig(
            history_limit=history_limit, cognitive_enabled=True
        ))
        for observation in observations:
            agent.observe(observation)
        result = agent.step(query)
        return result.text, len(agent._prompt(query)), 0, 0, True, False

    keys = target_keys(query)
    if condition == "append_memory":
        store = build_append_memory(observations)
        items = append_retrieve(store, keys, history=history, limit=memory_limit)
        stale = task == "supersession_current" and any(
            isinstance(item.content, dict) and item.content.get("kind") == "fact"
            for item in items
        )
        context = "\n".join(
            f"- {item.content.get('key')}: {item.content.get('value')}"
            for item in items
        ) or "NO_RELEVANT_MEMORY"
        prompt = f"RETRIEVED_MEMORY=\n{context}\n\n{query}"
        return backend.generate(prompt), len(prompt), len(store.items), len(items), bool(items), stale

    store = build_temporal_memory(observations)
    context = render_temporal(store, keys, history, memory_limit)
    prompt = f"RETRIEVED_MEMORY=\n{context}\n\n{query}"

    if condition == "temporal_memory":
        raw = backend.generate(prompt)
        return raw, len(prompt), len(store.store.items), len(temporal_retrieve(
            store, keys, history=history, limit=memory_limit
        )), bool(temporal_retrieve(store, keys, history=history, limit=memory_limit)), False

    agent = NSA(backend, config=RuntimeConfig(
        history_limit=history_limit,
        cognitive_enabled=True,
        memory_enabled=True,
        memory_limit=memory_limit,
    ))
    for observation in observations:
        match = FACT_RE.search(observation)
        if match:
            key, value = match.group(1).upper(), match.group(2).upper()
            kind = "update" if observation.upper().startswith("UPDATE:") else "fact"
            agent.remember(key, {"value": value, "key": key}, kind=kind)
        agent.observe(observation)

    # History uses the explicit temporal context directly. Current retrieval uses
    # the runtime's memory path so both representations are tested.
    if history:
        result = agent.step(prompt)
        model_prompt = agent._prompt(prompt)
    else:
        result = agent.step(query, memory_keys=keys)
        model_prompt = agent._prompt(query)

    runtime_items = temporal_retrieve(store, keys, history=history, limit=memory_limit)
    return result.text, len(model_prompt), len(store.store.items), len(runtime_items), bool(runtime_items), False


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "raw.jsonl"
    records: list[Record] = []
    total = (
        len(args.models) * len(args.seeds) * len(args.distractors)
        * args.trials * len(TASKS) * len(CONDITIONS)
    )
    done = 0

    with raw_path.open("a", encoding="utf-8") as handle:
        for model in args.models:
            backend = OllamaInferenceBackend(
                model=model,
                base_url=args.ollama_url,
            ) if args.ollama_url else OllamaInferenceBackend(model=model)
            for seed in args.seeds:
                for task in TASKS:
                    for distractors in args.distractors:
                        for trial in range(args.trials):
                            observations, query, expected, _, history_query = make_episode(
                                seed + trial, task, distractors
                            )
                            for condition in CONDITIONS:
                                done += 1
                                start = time.perf_counter()
                                response, prompt_chars, memory_items, retrieved_items, hit, stale = answer_for(
                                    condition, backend, observations, query, expected,
                                    task, args.history_limit, args.memory_limit
                                )
                                retrieval_correct, measured_stale = assess_memory_retrieval(
                                    condition, observations, query, expected, task, args.memory_limit
                                )
                                stale = stale or measured_stale
                                text = response.text if hasattr(response, "text") else str(response)
                                predicted = extract(text)
                                correct = (
                                    predicted == expected
                                    if "|" not in expected
                                    else text.strip().upper() == f"VALUE={expected}"
                                )
                                record = Record(
                                    model=model, seed=seed, condition=condition, task=task,
                                    distractors=distractors, trial=trial, expected=expected,
                                    predicted=predicted, correct=correct,
                                    memory_retrieval_correct=retrieval_correct,
                                    latency_seconds=round(time.perf_counter() - start, 4),
                                    prompt_chars=prompt_chars, response_chars=len(text),
                                    memory_items=memory_items, retrieved_items=retrieved_items,
                                    retrieval_hit=hit, stale_memory_retrieved=stale,
                                    history_query=history_query, response=text,
                                )
                                records.append(record)
                                handle.write(json.dumps(asdict(record)) + "\n")
                                handle.flush()
                                status = "OK" if correct else "FAIL"
                                retrieval_status = (
                                    "n/a" if retrieval_correct is None
                                    else ("MEM_OK" if retrieval_correct else "MEM_FAIL")
                                )
                                print(
                                    f"[{done}/{total}] {model} s={seed} {task} "
                                    f"d={distractors} {condition} -> {status} {retrieval_status} "
                                    f"({record.latency_seconds:.2f}s)",
                                    flush=True,
                                )

    summary = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.condition == condition]
        measured = [r for r in rows if r.memory_retrieval_correct is not None]
        summary[condition] = {
            "n": len(rows),
            "accuracy": statistics.mean(float(r.correct) for r in rows),
            "memory_retrieval_correct_rate": (
                statistics.mean(float(r.memory_retrieval_correct) for r in measured)
                if measured else None
            ),
            "retrieval_answer_gap": (
                statistics.mean(
                    float(r.memory_retrieval_correct) - float(r.correct)
                    for r in measured
                )
                if measured else None
            ),
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
            measured = [r for r in rows if r.memory_retrieval_correct is not None]
            by_task[task][condition] = {
                "n": len(rows),
                "accuracy": statistics.mean(float(r.correct) for r in rows),
                "memory_retrieval_correct_rate": (
                    statistics.mean(float(r.memory_retrieval_correct) for r in measured)
                    if measured else None
                ),
            }

    manifest = {
        "benchmark": "NSA Research Benchmark v4",
        "version": "4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "seeds": args.seeds,
        "trials": args.trials,
        "distractors": args.distractors,
        "history_limit": args.history_limit,
        "memory_limit": args.memory_limit,
        "conditions": list(CONDITIONS),
        "tasks": list(TASKS),
        "hypotheses": {
            "H13": "Temporal memory exposes explicit CURRENT versus SUPERSEDED semantics to the model.",
            "H14": "Memory retrieval correctness can be measured independently from final model-answer correctness.",
            "H15": "Historical retrieval remains correct when the previous value is explicitly labeled SUPERSEDED and the current value is provided as CURRENT_VALUE.",
            "H16": "Temporal memory retains its accuracy/latency advantage over cognitive-state-only context without requiring cognitive machinery.",
        },
        "scientific_boundary": (
            "Deterministic extraction isolates storage/retrieval semantics. "
            "v4 does not test autonomous LLM memory formation."
        ),
        "summary": summary,
        "by_task": by_task,
        "raw_artifact": str(raw_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen2.5:3b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    parser.add_argument("--distractors", nargs="+", type=int, default=[10, 50, 100, 250, 500, 1000])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=6)
    parser.add_argument("--memory-limit", type=int, default=3)
    parser.add_argument("--out", default="results/research-v4")
    parser.add_argument("--ollama-url", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
