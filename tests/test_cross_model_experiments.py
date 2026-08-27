"""Offline tests for the cross-model research harness."""
import json
from pathlib import Path
from experiments.cross_model.runner import ExperimentRunner, RunConfig
from experiments.cross_model.tasks import make_task

def test_task_generation_is_deterministic():
    for name in ("prediction","long_horizon","partial_observation","recovery","information_gain","authority"):
        assert make_task(name,7,8)==make_task(name,7,8)

def test_task_generation_changes_with_seed():
    assert make_task("long_horizon",1,8)!=make_task("long_horizon",2,8)

def test_aggregation_is_resumable(tmp_path: Path):
    p=tmp_path/"raw.jsonl"
    p.write_text(json.dumps({"model":"x","system":"raw","task":"prediction","seed":0,"horizon":4,"correct":True,"wall_time_s":1})+"\n")
    result=ExperimentRunner(RunConfig(("x",),output=str(tmp_path))).aggregate()
    assert result["records_total"]==1
    assert result["groups"]["x|raw|prediction|4"]["accuracy_mean"]==1.0

def test_score_exact_and_explanatory():
    assert ExperimentRunner._score("42",42)[0]
    assert ExperimentRunner._score("The answer is 42",42)[0]
    assert not ExperimentRunner._score("41",42)[0]
