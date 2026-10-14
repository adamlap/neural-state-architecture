"""NSA Research Benchmark v1: causal tests of explicit state utility."""
from __future__ import annotations
import argparse, json, re, statistics, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from nsa.agent import NSA, RuntimeConfig
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.episodes import GENERATORS, answer_instruction, facts as _facts, pool_size
from experiments.research_common.grading import lenient_correct

VALUE_RE = re.compile(r"VALUE\s*=\s*([A-Z0-9_-]+)", re.IGNORECASE)
CONDITIONS = ("raw", "transcript", "bounded", "nsa_no_state", "nsa_state")
TASKS = ("recall", "interference", "supersession")

@dataclass
class Record:
    model: str; seed: int; condition: str; task: str; delay: int; trial: int
    expected: str; predicted: str | None; correct: bool; latency_seconds: float
    input_chars_proxy: int; response_chars: int; state_enabled: bool
    history_limit: int; response: str
    correct_lenient: bool = False

def facts(seed: int, n: int = 24):
    return _facts(seed, n)

def make_episode(seed: int, task: str, delay: int, generator: str = 'unique', instruction: str = 'legacy'):
    # interference needs 2*delay distractors; the pool must not wrap onto the target key
    pairs = facts(seed, pool_size(2 * delay, generator, minimum=24, legacy=24)); key, value = pairs[0]
    observations = [f'FACT: {key} has value {value}.']
    expected = value
    if task == 'supersession':
        new = f'V{seed % 1000:03d}_CURRENT'
        observations = [f'FACT: {key} has value {value}.', f'UPDATE: {key} now has value {new}; the previous value is obsolete.']
        expected = new
    for i in range(delay):
        k, v = pairs[(i + 1) % len(pairs)]; observations.append(f'DISTRACTOR: {k} has value {v}.')
    if task == 'interference':
        for i in range(delay, delay * 2):
            k, v = pairs[(i + 2) % len(pairs)]; observations.append(f'DISTRACTOR: {k} has value {v}.')
    query = f'What is the current value of {key}? {answer_instruction(instruction)}'
    return observations, query, expected

def extract(text: str):
    m = VALUE_RE.search(text or ''); return m.group(1).upper() if m else None

def prompt_for(condition, observations, query, history_limit):
    if condition == 'raw': context = ''
    elif condition == 'transcript': context = '\n'.join(observations)
    else: context = '\n'.join(observations[-history_limit:])
    return ('You are a benchmark participant. Use only the supplied observations. Do not invent facts.\n' +
            (f'OBSERVATIONS:\n{context}\n' if context else '') + query)

def run_condition(backend, condition, observations, query, history_limit):
    started = time.perf_counter()
    if condition.startswith('nsa_'):
        enabled = condition == 'nsa_state'
        # nsa_no_state keeps the same bounded history window and drops only the state text
        # (previously it dropped history too, so it was really "no context at all").
        agent = NSA(backend, config=RuntimeConfig(history_limit=history_limit, include_state_in_prompt=enabled, include_history_in_prompt=True, cognitive_enabled=enabled))
        for obs in observations: agent.observe(obs, source='benchmark')
        result = agent.step(query)
        prompt = backend.last.prompt if getattr(backend, 'last', None) else ''
        return result.text, len(prompt), time.perf_counter() - started
    prompt = prompt_for(condition, observations, query, history_limit)
    result = backend.generate(prompt)
    return (result.text if hasattr(result, 'text') else str(result)), len(prompt), time.perf_counter() - started

def run(args):
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True); raw_path = root / 'raw.jsonl'; records = []
    total = len(args.models) * len(args.seeds) * len(TASKS) * len(args.delays) * args.trials * len(CONDITIONS)
    with raw_path.open('w', encoding='utf-8') as raw:
        for model in args.models:
            backend = FixedDecodingBackend(OllamaInferenceBackend(model_name=model, base_url=args.ollama_url), temperature=0.0, max_tokens=32)
            for seed in args.seeds:
                for task in TASKS:
                    for delay in args.delays:
                        for trial in range(args.trials):
                            observations, query, expected = make_episode(seed + trial * 100003, task, delay, args.generator, args.instruction)
                            for condition in CONDITIONS:
                                text, chars, latency = run_condition(backend, condition, observations, query, args.history_limit)
                                predicted = extract(text)
                                rec = Record(model, seed, condition, task, delay, trial, expected, predicted, predicted == expected, latency, chars, len(text), condition == 'nsa_state', args.history_limit, text[:2000], lenient_correct(text, expected))
                                records.append(rec); raw.write(json.dumps(asdict(rec), ensure_ascii=False) + '\n'); raw.flush(); sym = "✅" if rec.correct else "✗"; print(f"[{len(records)}/{total}] {model} s={seed} {task} d={delay} {condition} -> {sym} ({latency:.2f}s)", flush=True)
    def mean(xs): return statistics.fmean(xs) if xs else 0.0
    summary = {}
    for condition in CONDITIONS:
        rows = [r for r in records if r.condition == condition]
        summary[condition] = {'n': len(rows), 'accuracy': mean([float(r.correct) for r in rows]), 'accuracy_lenient': mean([float(r.correct_lenient) for r in rows]), 'latency_seconds_mean': mean([r.latency_seconds for r in rows]), 'input_chars_proxy_mean': mean([r.input_chars_proxy for r in rows]), 'response_chars_mean': mean([r.response_chars for r in rows])}
    cells = {}
    for task in TASKS:
        for delay in args.delays:
            for condition in CONDITIONS:
                rows = [r for r in records if r.task == task and r.delay == delay and r.condition == condition]
                cells[f'{task}/delay={delay}/{condition}'] = {'n': len(rows), 'accuracy': mean([float(r.correct) for r in rows]), 'accuracy_lenient': mean([float(r.correct_lenient) for r in rows])}
    summary['cells'] = cells
    def delta(a, b):
        return {'accuracy_delta': summary[a]['accuracy'] - summary[b]['accuracy'], 'latency_delta_seconds': summary[a]['latency_seconds_mean'] - summary[b]['latency_seconds_mean'], 'input_chars_delta': summary[a]['input_chars_proxy_mean'] - summary[b]['input_chars_proxy_mean']}
    result = {'benchmark':'NSA Research Benchmark v1','version':'1.2','generator':args.generator,'instruction':args.instruction,'decoding':{'temperature':0.0,'max_tokens':32,'num_ctx':32768},'generated_at_utc':datetime.now(timezone.utc).isoformat(),'models':args.models,'seeds':args.seeds,'trials':args.trials,'delays':args.delays,'history_limit':args.history_limit,'conditions':list(CONDITIONS),'hypotheses':{'H1':'nsa_state > nsa_no_state on delayed recall with matched bounded history','H2':'nsa_state degrades more slowly as irrelevant observations increase','H3':'nsa_state maintains supersession accuracy without increased stale-memory errors','H4':'accuracy gains must be interpreted jointly with input-size and latency proxies'},'summary':summary,'key_deltas':{'nsa_state_vs_nsa_no_state':delta('nsa_state','nsa_no_state'),'nsa_state_vs_bounded':delta('nsa_state','bounded'),'nsa_state_vs_transcript':delta('nsa_state','transcript')},'raw_artifact':str(raw_path),'scientific_boundary':'This suite tests explicit state utility in the current runtime. It does not establish consciousness, AGI, or general superiority.'}
    (root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8'); print(json.dumps(result, indent=2)); return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--models', nargs='+', default=['qwen2.5:3b']); p.add_argument('--seeds', nargs='+', type=int, default=[7,17,37]); p.add_argument('--delays', nargs='+', type=int, default=[2,6,12,20]); p.add_argument('--trials', type=int, default=2); p.add_argument('--history-limit', type=int, default=6); p.add_argument('--generator', choices=GENERATORS, default='unique'); p.add_argument('--instruction', choices=('legacy', 'plain'), default='plain'); p.add_argument('--out', default='results/research-v1.2'); p.add_argument('--ollama-url', default=None); run(p.parse_args())

if __name__ == '__main__': main()