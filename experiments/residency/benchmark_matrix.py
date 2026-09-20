"""Run repeatable cold/warm residency experiments for Qwen checkpoints.

If --model-path is omitted, the selected model is downloaded automatically
through Hugging Face Hub and reused from the local cache on subsequent runs.
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
from huggingface_hub import snapshot_download

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


def _resolve_model_path(model_key: str, requested_path: str | None) -> str:
    spec = get_local_model(model_key)
    if requested_path:
        path = Path(requested_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return spec.checkpoint_path({spec.env_var: str(path)})

    cached = Path(spec.resolve_path())
    if cached.exists():
        return spec.checkpoint_path()

    print(f"Model {spec.model_id} not found locally; downloading to the Hugging Face cache...")
    snapshot = snapshot_download(repo_id=spec.model_id)
    return snapshot


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


def run_condition(args: argparse.Namespace, prefetch: bool, model_path: str) -> list[dict[str, Any]]:
    rows = []
    for run in range(1, args.runs + 1):
        row = _run_once(
            model_key=args.model,
            model_path=model_path,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            prefetch=prefetch,
            vram_gb=args.vram_gb,
            ram_gb=args.ram_gb,
        )
        row.update(
            {
                "model": args.model,
                "model_path": str(Path(model_path).expanduser()),
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
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local checkpoint path. If omitted, download/reuse the Hugging Face checkpoint.",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vram-gb", type=float, default=None)
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--prefetch", choices=["on", "off", "both"], default="both")
    parser.add_argument(
        "--prompt",
        default="Explain how persistent cognitive state can improve an agent's reasoning.",
    )
    parser.add_argument("--output", type=Path, default=Path("results/residency/matrix.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("--runs and --max-tokens must be positive")

    spec = get_local_model(args.model)
    model_path = _resolve_model_path(args.model, args.model_path)
    vram_gb = args.vram_gb if args.vram_gb is not None else spec.vram_budget_gb
    ram_gb = args.ram_gb if args.ram_gb is not None else spec.ram_budget_gb

    conditions = [True, False] if args.prefetch == "both" else [args.prefetch == "on"]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows.extend(run_condition(args, condition, model_path))

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
            "model_id": spec.model_id,
            "model_path": str(Path(model_path).expanduser()),
            "downloaded_automatically": args.model_path is None,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "prefetch": args.prefetch,
            "vram_gb": vram_gb,
            "ram_gb": ram_gb,
            "prompt": args.prompt,
        },
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
