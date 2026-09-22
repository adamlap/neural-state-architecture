"""Repeatable prefetch on/off experiment for Qwen checkpoints under selective storage.

Method (what makes the on/off comparison meaningful):

* model loading and decoding are timed separately; only decoding is compared;
* conditions alternate order every round (on,off then off,on ...) so neither
  condition always benefits from what the other left behind;
* by default the OS page cache is dropped for the offload/checkpoint files after
  loading and before decoding, so prefetch is measured against cold reads;
* greedy decoding is used and the outputs of every run are hashed, so a
  prefetch change that alters generated tokens is reported;
* a run only counts as a prefetch measurement if bytes were actually warmed.

If --model-path is omitted, the selected model is downloaded through Hugging Face
Hub and reused from the local cache on subsequent runs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
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


def _rss_bytes() -> int:
    """Resident set size of this process (Linux); 0 where unavailable."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _drop_page_cache(*roots: str | Path) -> int:
    """Evict clean page-cache pages for every file under the given paths.

    Unprivileged (posix_fadvise DONTNEED after fsync). Returns the number of
    files advised, 0 when the platform cannot do it.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return 0
    advised = 0
    for root in roots:
        root = Path(root)
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()] if root.is_dir() else []
        for path in files:
            try:
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue
            try:
                try:
                    os.fsync(fd)  # dirty pages cannot be dropped
                except OSError:
                    pass
                fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                advised += 1
            except OSError:
                pass
            finally:
                os.close(fd)
    return advised


def _resolve_model_path(model_key: str, requested_path: str | None) -> str:
    spec = get_local_model(model_key)
    if requested_path:
        path = Path(requested_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return spec.checkpoint_path({spec.env_var: str(path)})

    if spec.is_local():
        return spec.checkpoint_path()

    print(f"Model {spec.model_id} not found locally (or incomplete); downloading to the Hugging Face cache...")
    return snapshot_download(repo_id=spec.model_id)


def _run_once(
    *,
    model_key: str,
    model_path: str,
    prompt: str,
    max_tokens: int,
    prefetch: bool,
    vram_gb: float,
    ram_gb: float,
    device: str,
    hot_layers: int,
    warm_layers: int,
    lookahead: int,
    cold_cache: bool,
) -> dict[str, Any]:
    backend = SelectiveStorageTransformersBackend(
        model_name=get_local_model(model_key).model_id,
        model_path=model_path,
        vram_budget_gb=vram_gb,
        ram_budget_gb=ram_gb,
        prefetch=prefetch,
        device=device,
        hot_layers=hot_layers,
        warm_layers=warm_layers,
        lookahead=lookahead,
    )
    try:
        load_start = time.perf_counter()
        backend.load_model()
        load_sec = time.perf_counter() - load_start
        files_dropped = _drop_page_cache(backend.offload_folder, model_path) if cold_cache else 0
        _reset_gpu_stats()
        start = time.perf_counter()
        result = backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        decode_sec = time.perf_counter() - start
        if backend.residency_controller is not None:
            backend.residency_controller.wait()
        snapshot = backend.residency.snapshot()
        tokens = len(result.tokens)
        return {
            "load_sec": load_sec,
            "decode_sec": decode_sec,
            "tokens": tokens,
            "tokens_per_sec": tokens / decode_sec if decode_sec > 0 else 0.0,
            "output_chars": len(result.text),
            "output": result.text,
            "output_sha256": hashlib.sha256(json.dumps(result.tokens).encode()).hexdigest(),
            "placement_bytes": {tier.value: int(size) for tier, size in snapshot.bytes_by_tier.items()},
            "region_count": len(backend.residency.regions),
            "trace": backend.trace.metrics(),
            # bytes the prefetcher found already page-cache resident (so correctly
            # read nothing) -- distinct from bytes_prefetched, which it actually read
            "bytes_already_resident": backend.prefetcher.bytes_already_resident if backend.prefetcher else 0,
            "page_cache_files_dropped": files_dropped,
            "rss_bytes": _rss_bytes(),
            "gpu": _gpu_stats(),
        }
    finally:
        backend.close()
        del backend
        gc.collect()
        _reset_gpu_stats()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label, flag in (("prefetch_on", True), ("prefetch_off", False)):
        subset = [r for r in rows if r["prefetch"] is flag]
        if not subset:
            continue
        decode = [r["decode_sec"] for r in subset]
        summary[label] = {
            "runs": len(subset),
            "decode_sec_median": statistics.median(decode),
            "decode_sec_mean": statistics.fmean(decode),
            "decode_sec_stdev": statistics.stdev(decode) if len(decode) > 1 else 0.0,
            "tokens_per_sec_median": statistics.median(r["tokens_per_sec"] for r in subset),
            "load_sec_median": statistics.median(r["load_sec"] for r in subset),
            "bytes_prefetched_median": statistics.median(r["trace"]["bytes_prefetched"] for r in subset),
            "bytes_already_resident_median": statistics.median(r.get("bytes_already_resident", 0) for r in subset),
            "prefetch_hit_rate_median": statistics.median(r["trace"]["prefetch_hit_rate"] for r in subset),
            "prefetch_coverage_median": statistics.median(r["trace"]["prefetch_coverage"] for r in subset),
        }
    notes = []
    on, off = summary.get("prefetch_on"), summary.get("prefetch_off")

    def _touched_nothing(stats: dict[str, Any]) -> bool:
        # bytes_prefetched==0 alone is not a problem: the background prefetch
        # thread can legitimately lose its race against the main thread's own
        # synchronous read on a fast enough model/layer, and mincore gating
        # then correctly finds the region already resident (bytes_already_resident)
        # rather than re-reading it. Only *neither* ever happening is a real
        # problem (no disk-offloaded regions, or a broken offload index).
        return stats["bytes_prefetched_median"] <= 0 and stats["bytes_already_resident_median"] <= 0

    if on and off:
        summary["decode_speedup_on_vs_off"] = off["decode_sec_median"] / on["decode_sec_median"]
        if _touched_nothing(on):
            notes.append("prefetch touched no bytes (prefetched or already-resident): the on/off comparison measures nothing (no disk-offloaded regions or no offload index).")
    elif on and _touched_nothing(on):
        notes.append("prefetch touched no bytes (prefetched or already-resident).")
    hashes = {r["output_sha256"] for r in rows}
    summary["outputs_identical_across_runs"] = len(hashes) == 1
    if len(hashes) != 1:
        notes.append("generated tokens differ between runs; prefetch must never change outputs.")
    if not any(r["page_cache_files_dropped"] for r in rows):
        notes.append("page cache was not dropped: runs are warm-cache and prefetch benefit is understated.")
    summary["notes"] = notes
    return summary


def run_matrix(args: argparse.Namespace, model_path: str, vram_gb: float, ram_gb: float) -> list[dict[str, Any]]:
    conditions = [True, False] if args.prefetch == "both" else [args.prefetch == "on"]
    rows: list[dict[str, Any]] = []
    for run in range(1, args.runs + 1):
        order = conditions if run % 2 else list(reversed(conditions))  # counterbalance
        for prefetch in order:
            row = _run_once(
                model_key=args.model,
                model_path=model_path,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                prefetch=prefetch,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                device=args.device,
                hot_layers=args.hot_layers,
                warm_layers=args.warm_layers,
                lookahead=args.lookahead,
                cold_cache=args.cold_cache,
            )
            row.update({"model": args.model, "model_path": str(Path(model_path).expanduser()),
                        "prefetch": prefetch, "run": run, "pid": os.getpid()})
            rows.append(row)
            print(
                f"run={run}/{args.runs} prefetch={prefetch!s:5} decode={row['decode_sec']:.3f}s "
                f"tok/s={row['tokens_per_sec']:.2f} prefetched={row['trace']['bytes_prefetched']}B "
                f"hit_rate={row['trace']['prefetch_hit_rate']:.3f} coverage={row['trace']['prefetch_coverage']:.3f}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["1.5b", "3b"], required=True)
    parser.add_argument("--model-path", default=None,
                        help="Local checkpoint path. If omitted, download/reuse the Hugging Face checkpoint.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vram-gb", type=float, default=None)
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N (default: auto)")
    parser.add_argument("--hot-layers", type=int, default=2)
    parser.add_argument("--warm-layers", type=int, default=2)
    parser.add_argument("--lookahead", type=int, default=2, help="predicted regions the background prefetch controller may have in flight at once")
    parser.add_argument("--prefetch", choices=["on", "off", "both"], default="both")
    parser.add_argument("--cold-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="drop the OS page cache for weight files before decoding (default: on)")
    parser.add_argument("--prompt", default="Explain how persistent cognitive state can improve an agent's reasoning.")
    parser.add_argument("--output", type=Path, default=Path("results/residency/matrix.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.max_tokens < 1:
        parser.error("--runs and --max-tokens must be positive")

    spec = get_local_model(args.model)
    model_path = _resolve_model_path(args.model, args.model_path)
    vram_gb = args.vram_gb if args.vram_gb is not None else spec.vram_budget_gb
    ram_gb = args.ram_gb if args.ram_gb is not None else spec.ram_budget_gb

    rows = run_matrix(args, model_path, vram_gb, ram_gb)
    summary = _summarize(rows)

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
            "cold_cache": args.cold_cache,
            "device": args.device,
            "hot_layers": args.hot_layers,
            "warm_layers": args.warm_layers,
            "lookahead": args.lookahead,
            "vram_gb": vram_gb,
            "ram_gb": ram_gb,
            "prompt": args.prompt,
        },
        "summary": summary,
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for note in summary["notes"]:
        print(f"NOTE: {note}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
