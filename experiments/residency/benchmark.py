"""Run a first selective-storage benchmark against local Qwen checkpoints.

Example:
  python -m experiments.residency.benchmark --models 1.5b,3b --root /models
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend

MODELS={
    "1.5b": ("Qwen/Qwen2.5-1.5B-Instruct","Qwen2.5-1.5B-Instruct"),
    "3b": ("Qwen/Qwen2.5-3B-Instruct","Qwen2.5-3B-Instruct"),
}

def run_one(model_key:str, root:Path, prompt:str, vram_gb:float, ram_gb:float)->dict:
    model_name,folder=MODELS[model_key]
    backend=SelectiveStorageTransformersBackend(
        model_name=model_name, model_path=str(root/folder),
        mode="cached", vram_budget_gb=vram_gb, ram_budget_gb=ram_gb,
    )
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    out=backend.generate(prompt,max_tokens=64,temperature=0.0)
    elapsed=time.perf_counter()-started
    backend.close()
    peak_vram=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    snap=backend.residency.snapshot()
    return {
        "model":model_name,"elapsed_sec":elapsed,
        "peak_vram_bytes":peak_vram,
        "resident_bytes":snap.bytes_by_tier,
        "regions":len(backend.residency.regions),
        "planned_scores":snap.scores,
        "output":out.text,
    }

def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",default="1.5b,3b")
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--prompt",default="Explain how persistent cognitive state can improve an agent's reasoning.")
    parser.add_argument("--vram-gb",type=float,default=4.0)
    parser.add_argument("--ram-gb",type=float,default=8.0)
    parser.add_argument("--output",type=Path,default=Path("results/residency/benchmark.json"))
    args=parser.parse_args()
    results=[run_one(k,args.root,args.prompt,args.vram_gb,args.ram_gb) for k in args.models.split(",")]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,default=str))
    print(json.dumps(results,indent=2,default=str))

if __name__=="__main__":
    main()
