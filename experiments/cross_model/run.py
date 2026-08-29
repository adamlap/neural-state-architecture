"""CLI entry point with smoke/local/reproduction profiles."""
from __future__ import annotations
import argparse
from .runner import ExperimentRunner, RunConfig, DEFAULT_TASKS

PROFILES={
 "smoke":dict(seeds=2,tasks=("prediction","long_horizon"),horizons=(4,8)),
 "local":dict(seeds=20,tasks=DEFAULT_TASKS,horizons=(4,8,16)),
 "reproduction":dict(seeds=50,tasks=DEFAULT_TASKS,horizons=(4,8,16,32)),
}

def main():
    p=argparse.ArgumentParser(description="NSA cross-model experiment suite")
    p.add_argument("--profile",choices=PROFILES,default="local"); p.add_argument("--models",required=True)
    p.add_argument("--output",default="results/cross_model"); p.add_argument("--host",default="http://127.0.0.1:11434"); p.add_argument("--timeout",type=float,default=300)
    a=p.parse_args(); d=PROFILES[a.profile]
    cfg=RunConfig(tuple(x.strip() for x in a.models.split(',') if x.strip()),output=a.output,host=a.host,timeout=a.timeout,**d)
    ExperimentRunner(cfg).run()

if __name__=="__main__": main()
