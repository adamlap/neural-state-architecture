"""Statistical aggregation and report generation for completed runs."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

def analyze(path: str) -> dict:
    p=Path(path); rows=[json.loads(x) for x in (p/"raw.jsonl").read_text().splitlines() if x.strip()]
    groups={}
    for r in rows:
        if "correct" not in r: continue
        groups.setdefault((r["model"],r["system"],r["task"],r["horizon"]),[]).append(r)
    out={}
    for k,rs in groups.items():
        x=[int(r["correct"]) for r in rs]; n=len(x); mean=sum(x)/n
        se=math.sqrt(mean*(1-mean)/n) if n else 0
        ci=1.96*se
        out["|".join(map(str,k))]={"n":n,"accuracy":mean,"ci95_low":max(0,mean-ci),"ci95_high":min(1,mean+ci),"wall_time_mean_s":statistics.mean(r.get("wall_time_s",0) for r in rs)}
    deltas={}
    for model in sorted({r["model"] for r in rows}):
        for task in sorted({r["task"] for r in rows}):
            for h in sorted({r["horizon"] for r in rows if r["task"]==task}):
                def val(system):
                    z=out.get(f"{model}|{system}|{task}|{h}"); return z["accuracy"] if z else None
                raw=val("raw")
                if raw is not None:
                    for system in ("memory","nsa","nsa_cce","nsa_cce_governance"):
                        v=val(system)
                        if v is not None: deltas[f"{model}|{task}|{h}|{system}-raw"]=v-raw
    result={"schema_version":1,"groups":out,"deltas_vs_raw":deltas,"interpretation":"Descriptive results only; confidence intervals are Wald binomial intervals. Do not infer causality or generalization from this report alone."}
    (p/"statistics.json").write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
    lines=["# NSA Cross-Model Experiment Report","","## Results","","| Model | System | Task | Horizon | N | Accuracy | 95% CI |","|---|---|---|---:|---:|---:|---|"]
    for k,v in sorted(out.items()):
        model,system,task,h=k.split("|"); lines.append(f"| {model} | {system} | {task} | {h} | {v['n']} | {v['accuracy']:.3f} | [{v['ci95_low']:.3f}, {v['ci95_high']:.3f}] |")
    lines += ["","## Interpretation","","Positive deltas are exploratory evidence, not proof of superiority. Inspect per-seed raw data and run independent replications before making research claims."]
    (p/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="results/cross_model"); args=ap.parse_args(); print(json.dumps(analyze(args.input),indent=2))
if __name__=="__main__": main()
