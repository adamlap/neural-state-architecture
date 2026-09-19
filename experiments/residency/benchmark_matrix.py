"""Run repeatable cold/warm residency experiments for local Qwen checkpoints.

This harness intentionally reports measurements rather than declaring a winner.
It compares the current NSA selective-storage backend with prefetch disabled/enabled
and records residency telemetry, generation latency, and GPU memory.

Example:
  python -m experiments.residency.benchmark_matrix \
    --model 1.5b --model-path /models/Qwen2.5-1.5B-Instruct \
    --runs 3 --max-tokens 64 --prefetch both

The script does not download models.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend
from nsa.runtime.inference.model_registry import get_local_model


def _gpu_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _reset_gpu_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_once(
    *,
    model_key: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    prefetch: bool,
    vram_gb: float,
    ram_gb: float,
) -> dict[str, Any]:
    backend = SelectiveStorageTransformersBackend(
        model_name=get_local_model(model_key).model_id,
        model_path=model_path,
        vram_budget_gb=vram_gb,
        ram_budget_gb=ram_gb,
        prefetch=prefetch,
    )
    try:
        _reset_gpu_stats()
        start = time.perf_counter()
        result = backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - start
        snapshot = backend.residency.snapshot()
        trace = backend.trace.metrics()
        return {
            "elapsed_sec": elapsed,
            "output_chars": len(result.text),
            "output": result.text,
            "resident_bytes": {str(k): int(v) for k, v in snapshot.bytes_by_tier.items()},
            "region_count": len(backend.residency.regions),
            "trace": trace,
            "gpu": _gpu_stats(),
        }
    finally:
        backend.close()
        gc.collect()
        _reset_gpu_stats()


def run_condition(args: argparse.Namespace, prefetch: bool) -> list[dict[str, Any]]:
    rows = []
    for run in range(1, args.runs + 1):
        row = _run_once(
            model_key=args.model,
            model_path=args.model_path,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            prefetch=prefetch,
            vram_gb=args.vram_gb,
            ram_gb=args.ram_gb,
        )
        row.update(
            {
                "model": args.model,
                "model_path": str(Path(args.model_path).expanduser()),
                "prefetch": prefetch,
                "run": run,
                "pid": os.getpid(),
            }
        )
        rows.append(row)
        print(
            f"prefetch={prefetch} run={run}/{args.runs} "
            f"elapsed={row['elapsed_sec']:.3f}s "
            f"prefetch_hits={row['trace']['prefetch_hits']} "
            f"hit_rate={row['trace']['prefetch_hit_rate']:.3f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["1.5b", "3b"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vram-gb", type=float, default=4.0)
    parser.add_argument("--ram-gb", type=float, default=8.0)
    parser.add_argument(
        "--prefetch",
        choices=["on", "off", "both"],
        default="both",
        help="Run predictive page-cache prefetch on, off, or both.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how persistent cognitive state can improve an agent's reasoning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/residency/matrix.json"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("--runs and --max-tokens must be positive")

    conditions = [True, False] if args.prefetch == "both" else [args.prefetch == "on"]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows.extend(run_condition(args, condition))

    payload = {
        "experiment": "nsa_selective_storage_matrix",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": {
            "model": args.model,
            "model_path": str(Path(args.model_path).expanduser()),
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "prefetch": args.prefetch,
            "vram_gb": args.vram_gb,
            "ram_gb": args.ram_gb,
            "prompt": args.prompt,
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
