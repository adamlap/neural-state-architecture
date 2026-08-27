"""One-command, resumable cross-model experiment harness."""
from __future__ import annotations
import argparse,json,platform,socket,statistics,time
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
from .runners import Ollama,RunnerFactory
from .tasks import make_task
SYSTEMS=("raw","memory","nsa","nsa_cce","nsa_cce_governance")
DEFAULT_TASKS=("prediction","long_horizon","partial_observation","recovery","information_gain","authority")
@dataclass(frozen=True)
class RunConfig:
    models:tuple[str,...]; seeds:int=20; tasks:tuple[str,...]=DEFAULT_TASKS; horizons:tuple[int,...]=(4,8,16); output:str="results/cross_model"; host:str="http://127.0.0.1:11434"; timeout:float=300.0
class ExperimentRunner:
    def __init__(self,config:RunConfig):self.config=config
    def _record_path(self):
        p=Path(self.config.output);p.mkdir(parents=True,exist_ok=True);return p/"raw.jsonl"
    def _completed(self):
        done=set();p=self._record_path()
        if not p.exists():return done
        for line in p.read_text().splitlines():
            try:
                r=json.loads(line);done.add((r["model"],r["system"],r["task"],r["seed"],r["horizon"]))
            except (ValueError,KeyError):pass
        return done
    @staticmethod
    def _score(text,answer):
        got=" ".join(text.strip().lower().split()).strip("`*_.,!?\"'");expected=str(answer).strip().lower()
        return (got==expected or (got.split() and got.split()[-1]==expected)),got
    def run(self):
        if not self.config.models:raise ValueError("no models configured")
        path=self._record_path();done=self._completed();records=0
        for model_name in self.config.models:
            backend=Ollama(model_name,self.config.host,self.config.timeout)
            if not backend.available():print(f"SKIP {model_name}: Ollama unavailable");continue
            for task_name in self.config.tasks:
                for horizon in self.config.horizons:
                    for seed in range(self.config.seeds):
                        key_base=(model_name,task_name,seed,horizon)
                        task=make_task(task_name,seed,horizon)
                        # Every cell gets fresh state: no cross-seed/task contamination.
                        for system in SYSTEMS:
                            key=key_base+(system,)
                            if (model_name,system,task_name,seed,horizon) in done:continue
                            runner={r.name:r for r in RunnerFactory(backend).all()}[system]
                            try:
                                g=runner.run(task.prompt,protected=task.metadata.get("protected"));correct,norm=self._score(g.text,task.answer)
                                record={"model":model_name,"family":model_name.split(":",1)[0],"system":system,"task":task_name,"seed":seed,"horizon":horizon,"correct":correct,"answer":str(task.answer),"response":g.text,"normalized_response":norm,"input_chars":g.input_chars,"output_chars":g.output_chars,"model_calls":g.calls,"wall_time_s":g.latency_s,"blocked":g.blocked,"decision":g.decision,"metadata":task.metadata,"timestamp":time.time()}
                            except Exception as exc:
                                record={"model":model_name,"system":system,"task":task_name,"seed":seed,"horizon":horizon,"correct":False,"error":f"{type(exc).__name__}: {exc}","timestamp":time.time()}
                            with path.open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
                            records+=1;print(f"{model_name:18} {system:20} {task_name:20} h={horizon:2} seed={seed:3} {'OK' if record.get('correct') else 'FAIL'}")
        return self.aggregate(records)
    def aggregate(self,new_records=0):
        p=self._record_path();rows=[]
        for line in p.read_text().splitlines() if p.exists() else []:
            try:rows.append(json.loads(line))
            except ValueError:pass
        groups={}
        for r in rows:
            if "correct" not in r:continue
            groups.setdefault((r["model"],r["system"],r["task"],r["horizon"]),[]).append(r)
        aggregate={}
        for k,rs in groups.items():
            x=[int(r["correct"]) for r in rs];times=[r.get("wall_time_s",0) for r in rs];m=statistics.mean(x)
            aggregate["|".join(map(str,k))]={"n":len(x),"accuracy_mean":m,"accuracy_stdev":statistics.stdev(x) if len(x)>1 else 0.0,"wall_time_mean_s":statistics.mean(times),"correct":sum(x),"blocked":sum(bool(r.get("blocked")) for r in rs)}
        out={"schema_version":1,"systems":SYSTEMS,"config":asdict(self.config),"environment":{"python":platform.python_version(),"platform":platform.platform(),"hostname":socket.gethostname()},"records_total":len(rows),"records_added":new_records,"groups":aggregate}
        (Path(self.config.output)/"aggregate.json").write_text(json.dumps(out,indent=2,sort_keys=True),encoding="utf-8");return out
def parse_args():
    ap=argparse.ArgumentParser();ap.add_argument("--models",required=True);ap.add_argument("--seeds",type=int,default=20);ap.add_argument("--tasks",default=','.join(DEFAULT_TASKS));ap.add_argument("--horizons",default="4,8,16");ap.add_argument("--output",default="results/cross_model");ap.add_argument("--host",default="http://127.0.0.1:11434");ap.add_argument("--timeout",type=float,default=300);a=ap.parse_args();return RunConfig(tuple(x.strip() for x in a.models.split(',') if x.strip()),a.seeds,tuple(x.strip() for x in a.tasks.split(',') if x.strip()),tuple(int(x) for x in a.horizons.split(',')),a.output,a.host,a.timeout)
def main():
    c=parse_args();r=ExperimentRunner(c).run();print(json.dumps({"records_total":r["records_total"],"output":c.output},indent=2))
if __name__=="__main__":main()
