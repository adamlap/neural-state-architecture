"""NSA Research Benchmark v7: knowing what is not known (evidence-gated abstention).

Half of the questions are answerable from the event stream; half ask about a key that never
appeared. The instruction allows ``VALUE=UNKNOWN``. Conditions:

* ``transcript``     - full event log; the model decides whether it knows (generic escape clause).
* ``memory_generic`` - canonical memory retrieval is rendered (``NO_RELEVANT_MEMORY`` when empty); generic
                       escape clause ("if the value is not in the information provided, reply VALUE=UNKNOWN").
* ``memory_keyed``   - same retrieval; the escape clause is keyed to the retrieval result
                       ("only if RETRIEVED_MEMORY is exactly NO_RELEVANT_MEMORY, reply VALUE=UNKNOWN").
* ``state_gated``    - the runtime's soft state carries uncertainty derived from retrieval coverage; above the
                       threshold the system abstains *without calling the model*; otherwise the model gets a
                       plain instruction with no escape clause.

The two prompt-based memory conditions exist because abstention by instruction is wording- and model-dependent
(measured: qwen2.5:1.5b abstains even on present facts under the generic clause; 3b abstains less under the keyed one).

Metrics: answerable accuracy, hallucination rate on unanswerable questions (any answer other than
UNKNOWN), abstention on answerable questions (over-abstention), latency.

Scientific boundary: ``state_gated`` abstains on a deterministic rule (empty retrieval => uncertainty 1.0),
so its zero hallucination rate is structural. The empirical question is how far the ungated conditions are from it
(do small models say UNKNOWN when told memory is empty?) and what it saves in compute. The uncertainty signal is
retrieval coverage, not a learned or self-reported confidence; calibrating a signal for genuinely ambiguous evidence is not tested.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nsa import NSA, RuntimeConfig
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.episodes import facts, pool_size
from experiments.research_common.grading import extract, lenient_correct
from experiments.research_common.stats import holm, mcnemar_exact, wilson

VERSION = "7.0"
DECODING = {"temperature": 0.0, "max_tokens": 32, "num_ctx": 32768}
CONDITIONS = ("transcript", "memory_generic", "memory_keyed", "state_gated")
KINDS = ("recall", "supersession", "unanswerable")
ABSTAIN_ABOVE = 0.5
BASE = (
    "Reply with exactly one line: VALUE= followed by the value token and nothing else (example: VALUE=V123_456). "
    "The value token never includes VERSION or STATUS."
)
INSTRUCTION = {
    "generic": BASE + " If the value is not in the information provided, reply VALUE=UNKNOWN.",
    "keyed": BASE + " Only if RETRIEVED_MEMORY is exactly NO_RELEVANT_MEMORY, reply VALUE=UNKNOWN instead.",
    "none": BASE,
}
CONDITION_INSTRUCTION = {"transcript": "generic", "memory_generic": "generic", "memory_keyed": "keyed", "state_gated": "none"}


def make_episode(seed: int, kind: str, distractors: int):
    pairs = facts(seed, pool_size(distractors))
    key, old = pairs[0]
    lines = [f"FACT: {key} has value {old}."]
    expected = old
    if kind == "supersession":
        expected = f"V{seed % 1000:03d}_CURRENT"
        lines.append(f"UPDATE: {key} now has value {expected}; the previous value is obsolete.")
    lines += [f"DISTRACTOR: {k} has value {v}." for k, v in pairs[1:distractors + 1]]
    if kind == "unanswerable":
        key, expected = pairs[-1][0], "UNKNOWN"          # a key the stream never mentions
    return lines, f"What is the current value of {key}?", expected, key


_FACT = re.compile(r"(FACT|UPDATE|DISTRACTOR):\s+(ITEM_[0-9A-Z_]+)\s+(?:now\s+)?has value ([A-Z0-9_-]+)", re.IGNORECASE)


def build_agent(lines: list[str], backend) -> NSA:
    agent = NSA(backend, config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False))
    for line in lines:
        m = _FACT.search(line)
        if m:
            agent.remember(m.group(2).upper(), {"value": m.group(3).upper()}, kind="update" if m.group(1).upper() == "UPDATE" else "fact")
    return agent


def retrieval_uncertainty(agent: NSA, key: str) -> float:
    """Uncertainty from evidence coverage: 0 when memory holds the key, 1 when it holds nothing."""
    return 0.0 if agent.recall([key]) else 1.0


def answer_for(condition: str, backend, lines: list[str], question: str, key: str):
    """Return ``(text, model_called, uncertainty)``."""
    query = f"{question} {INSTRUCTION[CONDITION_INSTRUCTION[condition]]}"
    if condition == "transcript":
        return backend.generate("\n".join(lines) + "\n\n" + query).text, True, None
    agent = build_agent(lines, backend)
    uncertainty = retrieval_uncertainty(agent, key)
    if condition == "state_gated" and uncertainty > ABSTAIN_ABOVE:
        return "VALUE=UNKNOWN", False, uncertainty
    context = agent.memory.render([key])
    return backend.generate(f"RETRIEVED_MEMORY=\n{context}\n\n{query}").text, True, uncertainty


@dataclass
class Record:
    model: str
    seed: int
    condition: str
    kind: str
    distractors: int
    trial: int
    expected: str
    answered_unknown: bool
    correct: bool
    hallucinated: bool
    model_called: bool
    uncertainty: float | None
    latency_seconds: float
    timed_out: bool
    response: str


def grade(kind: str, expected: str, text: str, timed_out: bool) -> tuple[bool, bool, bool]:
    token = extract(text)
    unknown = (token == "UNKNOWN") or ("unknown" in (text or "").lower() and token is None)
    if kind == "unanswerable":
        return unknown, unknown, (not unknown) and not timed_out
    return unknown, (not timed_out) and lenient_correct(text, expected), False


def _cell(rows: list[Record]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    ans = [r for r in rows if r.kind != "unanswerable"]
    un = [r for r in rows if r.kind == "unanswerable"]
    out = {"n": n, "latency_seconds_mean": sum(r.latency_seconds for r in rows) / n,
           "model_call_rate": sum(r.model_called for r in rows) / n, "timeouts": sum(r.timed_out for r in rows)}
    if ans:
        k = sum(r.correct for r in ans)
        out |= {"answerable_n": len(ans), "answerable_accuracy": k / len(ans), "answerable_ci95": list(wilson(k, len(ans))),
                "over_abstention_rate": sum(r.answered_unknown for r in ans) / len(ans)}
    if un:
        h = sum(r.hallucinated for r in un)
        out |= {"unanswerable_n": len(un), "hallucination_rate": h / len(un), "hallucination_ci95": list(wilson(h, len(un)))}
    return out


def summarize(records: list[Record]) -> dict:
    out = {"by_condition": {c: _cell([r for r in records if r.condition == c]) for c in CONDITIONS}, "paired_tests": []}
    for metric, kinds in (("correct", ("recall", "supersession")), ("hallucinated", ("unanswerable",))):
        data = {c: {(r.model, r.seed, r.kind, r.distractors, r.trial): getattr(r, metric)
                    for r in records if r.condition == c and r.kind in kinds} for c in CONDITIONS}
        rows = []
        for a, b in (("state_gated", "memory_generic"), ("state_gated", "memory_keyed"), ("state_gated", "transcript"),
                     ("memory_keyed", "memory_generic")):
            if data[a] and data[b]:
                rows.append({"metric": metric, "first": a, "second": b, **mcnemar_exact(data[a], data[b])})
        for row, adj in zip(rows, holm([x["p"] for x in rows])):
            row["p_holm"] = adj
        out["paired_tests"] += rows
    return out


def _load(path: Path) -> list[Record]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(Record(**json.loads(line)))
            except Exception:
                continue
    return out


def write_manifest(root: Path, args, records: list[Record]) -> dict:
    manifest = {
        "benchmark": "NSA Research Benchmark v7 (evidence-gated abstention)", "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "models": args.models, "seeds": args.seeds,
        "trials": args.trials, "distractors": args.distractors, "decoding": DECODING, "abstain_above": ABSTAIN_ABOVE,
        "instructions": INSTRUCTION, "condition_instruction": CONDITION_INSTRUCTION,
        "conditions": list(CONDITIONS), "kinds": list(KINDS), "episodes_recorded": len(records),
        "hypotheses": {
            "H24": "Small models fabricate an answer for an unobserved key in a substantial share of episodes, with or without being told memory is empty.",
            "H25": "State-gated abstention removes fabrication without reducing answerable accuracy and avoids the model call when evidence is absent.",
        },
        "scientific_boundary": (
            "state_gated abstains on a deterministic rule, so its zero hallucination rate is structural; the uncertainty signal is retrieval "
            "coverage, not a calibrated confidence. Measured: how far ungated conditions are from it, over-abstention, and compute saved."
        ),
        **summarize(records), "raw_artifact": str(root / "raw.jsonl"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    raw, sidecar = root / "raw.jsonl", root / "config.json"
    signature = {"version": VERSION, "decoding": DECODING, "abstain_above": ABSTAIN_ABOVE, "instructions": INSTRUCTION}
    if getattr(args, "summarize_only", False):
        return write_manifest(root, args, _load(raw))
    if sidecar.exists() and json.loads(sidecar.read_text()) != signature:
        raise SystemExit(f"{sidecar} was written with a different configuration; use a new --out")
    sidecar.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")
    records = _load(raw)
    done = {(r.model, r.seed, r.condition, r.kind, r.distractors, r.trial) for r in records}
    total = len(args.models) * len(args.seeds) * len(args.distractors) * args.trials * len(KINDS) * len(CONDITIONS)
    n = len(records)
    with raw.open("a", encoding="utf-8") as handle:
        for model in args.models:
            inner = OllamaInferenceBackend(model=model, base_url=args.ollama_url, timeout_sec=args.timeout) \
                if args.ollama_url else OllamaInferenceBackend(model=model, timeout_sec=args.timeout)
            backend = FixedDecodingBackend(inner, **DECODING)
            for seed in args.seeds:
                for kind in KINDS:
                    for d in args.distractors:
                        for trial in range(args.trials):
                            lines, question, expected, key = make_episode(seed + trial * 100003, kind, d)
                            for condition in CONDITIONS:
                                if (model, seed, condition, kind, d, trial) in done:
                                    continue
                                n += 1
                                start = time.perf_counter()
                                backend.last = None
                                text, called, uncertainty = answer_for(condition, backend, lines, question, key)
                                timed_out = bool(backend.last and backend.last.timed_out)
                                unknown, correct, halluc = grade(kind, expected, text, timed_out)
                                rec = Record(model, seed, condition, kind, d, trial, expected, unknown, correct, halluc, called,
                                             uncertainty, round(time.perf_counter() - start, 3), timed_out, text)
                                records.append(rec)
                                handle.write(json.dumps(asdict(rec)) + "\n")
                                handle.flush()
                                print(f"[{n}/{total}] {model} s={seed} {kind} d={d} {condition} -> "
                                      f"{'HALLUCINATED' if halluc else ('UNKNOWN' if unknown else ('OK' if correct else 'WRONG'))} "
                                      f"({rec.latency_seconds:.2f}s)", flush=True)
    manifest = write_manifest(root, args, records)
    print(json.dumps({c: {k: v.get(k) for k in ("answerable_accuracy", "hallucination_rate", "model_call_rate")}
                      for c, v in manifest["by_condition"].items()}, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["qwen2.5:1.5b"])
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 37])
    p.add_argument("--distractors", nargs="+", type=int, default=[10, 250])
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--out", default="results/research-v7")
    p.add_argument("--ollama-url", default=None)
    run(p.parse_args())


if __name__ == "__main__":
    main()
