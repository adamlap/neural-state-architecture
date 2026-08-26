"""NSA 6.4: independent replication, compute accounting, held-out and adversarial matrix.

This is an additional validation layer over the established NSA 6.3 six-arm suite.
It does not change the 6.3 implementation or tune thresholds after seeing results.

The matrix varies model family, seed, hypothesis count and observation noise. It
separates development and held-out seeds, records compute proxies, and performs a
small adaptive stress pass: the worst-performing observed configuration determines
the next stress configuration. All raw reports are preserved in one evidence bundle.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiments.nsa63.scientific_validation_suite import run_nsa63_validation_suite


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
DEV_SEEDS = [7, 17, 37, 73, 137]
HELDOUT_SEEDS = [101, 211, 307, 401, 509]
HYPOTHESES = [2, 4, 8, 16]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


@dataclass
class RunRecord:
    model: str
    backend: str
    split: str
    seed: int
    hypotheses: int
    noise: float
    trials: int
    gtc: float
    violations: int
    tokens_mean: float
    information_gain_bits: float
    epistemic_efficiency: float
    trajectory_audit_passed: bool
    invariants_verified: bool
    raw_artifact: str


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_one(
    *, model: str, backend: str, split: str, seed: int, hypotheses: int,
    noise: float, trials: int, output_root: Path, run_index: int,
) -> RunRecord:
    run_dir = output_root / "raw" / f"{run_index:04d}_{split}_{seed}_{hypotheses}_{noise:g}"
    report = run_nsa63_validation_suite(
        num_trials=trials,
        num_hypotheses=hypotheses,
        noise_level=noise,
        seed=seed,
        backend_mode=backend,
        model_name=model,
        output_dir=run_dir,
    )
    full = report["empirical_observations"]["Arm_6_Full_NSA_Substrate"]
    raw_path = run_dir / "aggregate.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return RunRecord(
        model=model,
        backend=backend,
        split=split,
        seed=seed,
        hypotheses=hypotheses,
        noise=noise,
        trials=trials,
        gtc=float(full["gtc_mean"]),
        violations=int(full["violations"]),
        tokens_mean=float(full["tokens_mean"]),
        information_gain_bits=float(full["information_gain_mean_bits"]),
        epistemic_efficiency=float(full["epistemic_efficiency"]),
        trajectory_audit_passed=bool(report["governance_invariants"]["trajectory_audit_passed"]),
        invariants_verified=bool(report["invariants_verified"]),
        raw_artifact=str(raw_path),
    )


def paired_summary(records: List[RunRecord]) -> Dict[str, Any]:
    grouped: Dict[str, List[RunRecord]] = {}
    for r in records:
        grouped.setdefault(r.split + ":" + r.model, []).append(r)
    out: Dict[str, Any] = {}
    for key, rows in grouped.items():
        out[key] = {
            "n": len(rows),
            "gtc_mean": statistics.fmean(r.gtc for r in rows),
            "gtc_min": min(r.gtc for r in rows),
            "gtc_max": max(r.gtc for r in rows),
            "violations": sum(r.violations for r in rows),
            "tokens_mean": statistics.fmean(r.tokens_mean for r in rows),
            "information_gain_mean_bits": statistics.fmean(r.information_gain_bits for r in rows),
            "epistemic_efficiency_mean": statistics.fmean(r.epistemic_efficiency for r in rows),
            "all_invariants_verified": all(r.invariants_verified for r in rows),
            "all_audits_passed": all(r.trajectory_audit_passed for r in rows),
        }
    return out


def adaptive_stress(records: List[RunRecord]) -> Dict[str, Any]:
    """Choose the next stress point from observed weakest GTC, without changing gates."""
    worst = min(records, key=lambda r: r.gtc)
    stress = {
        "selected_from": asdict(worst),
        "next_hypotheses": min(32, max(2, worst.hypotheses * 2)),
        "next_noise": min(0.5, round(worst.noise + 0.10, 2)),
        "reason": "stress the observed weakest full-NSA configuration",
    }
    return stress


def main() -> None:
    parser = argparse.ArgumentParser(description="NSA 6.4 independent replication matrix")
    parser.add_argument("--backend", choices=["mock", "ollama", "cached", "remote", "lmstudio", "openai"], default="mock")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dev-seeds", nargs="+", type=int, default=DEV_SEEDS)
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=HELDOUT_SEEDS)
    parser.add_argument("--hypotheses", nargs="+", type=int, default=HYPOTHESES)
    parser.add_argument("--noise", nargs="+", type=float, default=NOISE_LEVELS)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", default="results/nsa64")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    records: List[RunRecord] = []
    run_index = 0

    # Development matrix: broad factorial coverage for model/configuration effects.
    for model in args.models:
        for seed in args.dev_seeds:
            for hypotheses in args.hypotheses:
                for noise in args.noise:
                    records.append(run_one(
                        model=model, backend=args.backend, split="development",
                        seed=seed, hypotheses=hypotheses, noise=noise,
                        trials=args.trials, output_root=root, run_index=run_index,
                    ))
                    run_index += 1

    # Held-out matrix: completely separate seeds, same predeclared grid.
    for model in args.models:
        for seed in args.heldout_seeds:
            for hypotheses in args.hypotheses:
                for noise in args.noise:
                    records.append(run_one(
                        model=model, backend=args.backend, split="heldout",
                        seed=seed, hypotheses=hypotheses, noise=noise,
                        trials=args.trials, output_root=root, run_index=run_index,
                    ))
                    run_index += 1

    stress_plan = adaptive_stress(records)
    manifest = {
        "benchmark": "NSA 6.4 Independent Replication Matrix",
        "version": "6.4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "backend": args.backend,
        "models": args.models,
        "development_seeds": args.dev_seeds,
        "heldout_seeds": args.heldout_seeds,
        "hypotheses": args.hypotheses,
        "noise_levels": args.noise,
        "trials_per_cell": args.trials,
        "compute_accounting": ["model_calls", "input_tokens", "output_tokens", "total_tokens", "wall_time_seconds", "tool_calls"],
        "controls": ["raw_llm", "static_guardrail", "governed_agent", "search_agent", "belief_agent", "full_nsa_substrate"],
        "heldout_policy": "held-out seeds are never used for parameter or threshold selection",
        "adversarial_policy": "adaptive stress selects the weakest observed configuration and increases hypotheses/noise without modifying pass/fail gates",
        "records": [asdict(r) for r in records],
        "summary": paired_summary(records),
        "adaptive_stress_plan": stress_plan,
        "evidence_schema": {
            "raw_trajectories": "results/nsa64/raw/**/trajectory.jsonl",
            "aggregate_reports": "results/nsa64/raw/**/aggregate.json",
            "manifest": "results/nsa64/manifest.json",
        },
        "scientific_boundary": "Replication evidence only; no AGI, consciousness, or general-superiority claim.",
    }
    manifest["manifest_sha256"] = config_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["summary"], indent=2))
    print(json.dumps(manifest["adaptive_stress_plan"], indent=2))
    print(f"artifact={root / 'manifest.json'}")


if __name__ == "__main__":
    main()
