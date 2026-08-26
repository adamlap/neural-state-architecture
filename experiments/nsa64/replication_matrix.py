"""NSA 6.4 independent replication, compute accounting, held-out and adaptive stress."""
from __future__ import annotations
import argparse, hashlib, json, platform, statistics, subprocess, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from experiments.nsa63.scientific_validation_suite import run_nsa63_validation_suite

DEFAULT_MODELS = ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-4B", "meta-llama/Llama-3.1-8B-Instruct"]
DEV_SEEDS = [7, 17, 37, 73, 137]
HELDOUT_SEEDS = [101, 211, 307, 401, 509]
HYPOTHESES = [2, 4, 8, 16]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]

@dataclass
class RunRecord:
    model: str; backend: str; split: str; seed: int; hypotheses: int; noise: float; trials: int
    gtc: float; violations: int; tokens_mean: float; information_gain_bits: float
    epistemic_efficiency: float; wall_time_seconds: float; model_calls: int; tool_calls: Optional[int]
    trajectory_steps: int; trajectory_audit_passed: bool; invariants_verified: bool; raw_artifact: str

def git_revision() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return "unknown"

def config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def run_one(*, model: str, backend: str, split: str, seed: int, hypotheses: int, noise: float,
            trials: int, output_root: Path, run_index: int) -> RunRecord:
    run_dir = output_root / "raw" / f"{run_index:04d}_{split}_{seed}_{hypotheses}_{noise:g}"
    started = time.perf_counter()
    report = run_nsa63_validation_suite(num_trials=trials, num_hypotheses=hypotheses, noise_level=noise,
                                        seed=seed, backend_mode=backend, model_name=model, output_dir=run_dir)
    wall_time = time.perf_counter() - started
    full = report["empirical_observations"]["Arm_6_Full_NSA_Substrate"]
    traj_path = run_dir / "trajectory.jsonl"
    full_steps = full_tokens = total_steps = 0
    if traj_path.exists():
        with traj_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                total_steps += 1
                try:
                    row = json.loads(line)
                    if row.get("arm") == "Arm_6_Full_NSA_Substrate":
                        full_steps += 1
                        full_tokens += int(row.get("tokens_consumed", 0))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
    raw_path = run_dir / "aggregate.json"
    report["nsa64_compute_accounting"] = {
        "wall_time_seconds": wall_time, "trajectory_steps_all_arms": total_steps,
        "full_nsa_trajectory_steps": full_steps, "full_nsa_model_calls_proxy": full_steps,
        "full_nsa_tokens_from_machine_trace": full_tokens, "tool_calls": None,
        "reported_full_nsa_mean_tokens": float(full["tokens_mean"]),
        "note": "model-call count is an arm-specific trajectory-step proxy; the NSA 6.3 logger does not expose an independent tool-call counter, so tool_calls is null",
    }
    raw_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return RunRecord(model, backend, split, seed, hypotheses, noise, trials, float(full["gtc_mean"]),
        int(full["violations"]), float(full["tokens_mean"]), float(full["information_gain_mean_bits"]),
        float(full["epistemic_efficiency"]), wall_time, full_steps, None, total_steps,
        bool(report["governance_invariants"]["trajectory_audit_passed"]), bool(report["invariants_verified"]), str(raw_path))

def paired_summary(records: List[RunRecord]) -> Dict[str, Any]:
    grouped: Dict[str, List[RunRecord]] = {}
    for r in records: grouped.setdefault(r.split + ":" + r.model, []).append(r)
    return {key: {"n": len(rows), "gtc_mean": statistics.fmean(r.gtc for r in rows),
        "gtc_min": min(r.gtc for r in rows), "gtc_max": max(r.gtc for r in rows),
        "violations": sum(r.violations for r in rows), "tokens_mean": statistics.fmean(r.tokens_mean for r in rows),
        "wall_time_mean_seconds": statistics.fmean(r.wall_time_seconds for r in rows),
        "model_calls_mean": statistics.fmean(r.model_calls for r in rows),
        "information_gain_mean_bits": statistics.fmean(r.information_gain_bits for r in rows),
        "epistemic_efficiency_mean": statistics.fmean(r.epistemic_efficiency for r in rows),
        "all_invariants_verified": all(r.invariants_verified for r in rows),
        "all_audits_passed": all(r.trajectory_audit_passed for r in rows)} for key, rows in grouped.items()}

def main() -> None:
    p = argparse.ArgumentParser(description="NSA 6.4 independent replication matrix")
    p.add_argument("--backend", choices=["mock", "ollama", "cached", "remote", "lmstudio", "openai"], default="mock")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS); p.add_argument("--dev-seeds", nargs="+", type=int, default=DEV_SEEDS)
    p.add_argument("--heldout-seeds", nargs="+", type=int, default=HELDOUT_SEEDS); p.add_argument("--hypotheses", nargs="+", type=int, default=HYPOTHESES)
    p.add_argument("--noise", nargs="+", type=float, default=NOISE_LEVELS); p.add_argument("--trials", type=int, default=20)
    p.add_argument("--out", default="results/nsa64"); args = p.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True); records: List[RunRecord] = []; run_index = 0
    for split, seeds in (("development", args.dev_seeds), ("heldout", args.heldout_seeds)):
        for model in args.models:
            for seed in seeds:
                for hypotheses in args.hypotheses:
                    for noise in args.noise:
                        records.append(run_one(model=model, backend=args.backend, split=split, seed=seed, hypotheses=hypotheses,
                                               noise=noise, trials=args.trials, output_root=root, run_index=run_index)); run_index += 1
    dev_records = [r for r in records if r.split == "development"]
    worst = min(dev_records, key=lambda r: r.gtc)
    stress_h = min(32, max(2, worst.hypotheses * 2)); stress_n = min(0.5, round(worst.noise + 0.10, 2))
    stress = run_one(model=worst.model, backend=args.backend, split="adversarial", seed=worst.seed + 10000,
                     hypotheses=stress_h, noise=stress_n, trials=args.trials, output_root=root, run_index=run_index)
    manifest = {"benchmark": "NSA 6.4 Independent Replication Matrix", "version": "6.4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(), "python": platform.python_version(),
        "backend": args.backend, "models": args.models, "development_seeds": args.dev_seeds, "heldout_seeds": args.heldout_seeds,
        "hypotheses": args.hypotheses, "noise_levels": args.noise, "trials_per_cell": args.trials,
        "controls": ["raw_llm", "static_guardrail", "governed_agent", "search_agent", "belief_agent", "full_nsa_substrate"],
        "compute_accounting": {"measured": ["wall_time_seconds", "full_nsa_trajectory_steps", "full_nsa_model_calls_proxy", "full_nsa_tokens_from_machine_trace"],
            "unavailable": ["tool_calls"], "policy": "Unavailable metrics remain null; no values are fabricated."},
        "heldout_policy": "held-out seeds are never used for stress selection or threshold selection",
        "adversarial_policy": "adaptive stress selects only from development, then increases hypotheses/noise without modifying gates",
        "records": [asdict(r) for r in records], "adversarial_record": asdict(stress), "summary": paired_summary(records),
        "adaptive_stress_selection": {"source": asdict(worst), "hypotheses": stress_h, "noise": stress_n},
        "scientific_boundary": "Replication evidence only; no AGI, consciousness, or general-superiority claim.",
        "evidence_schema": {"raw": "results/nsa64/raw/**", "manifest": "results/nsa64/manifest.json"}}
    manifest["manifest_sha256"] = config_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2)); print(json.dumps(manifest["adversarial_record"], indent=2)); print(f"artifact={root / 'manifest.json'}")

if __name__ == "__main__": main()
