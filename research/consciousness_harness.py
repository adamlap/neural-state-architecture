"""Reproducible consciousness-property research harness."""
from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence
from nsa.cognition.embodied import ActiveCognition, ActiveCognitionState
from nsa.cognition.substrate import CognitiveState, CognitiveSubstrate, CognitiveSwitches
THEORY_MECHANISMS: Mapping[str, tuple[str,...]]={"gwt":("workspace","attention","broadcast","ignition"),"iit_inspired":("integration","causal_influence"),"recurrent_processing":("recurrence","feedback"),"higher_order":("self_model","metacognition"),"predictive_processing":("prediction","prediction_error","uncertainty")}
@dataclass(frozen=True)
class ExperimentConfig:
    seed:int=0; theory:str="gwt"; steps:int=32; switches:CognitiveSwitches=CognitiveSwitches(); embodied:bool=False; model_name:str="deterministic"; environment_name:str="synthetic"
@dataclass(frozen=True)
class ExperimentObservation:
    step:int; observation:Any; action:str|None; metrics:Mapping[str,float]
@dataclass(frozen=True)
class ExperimentResult:
    config:ExperimentConfig; observations:tuple[ExperimentObservation,...]; summary:Mapping[str,float]
    def to_dict(self)->dict[str,Any]: return {"config":self.config.__dict__,"summary":dict(self.summary),"observations":[dict(step=o.step,observation=o.observation,action=o.action,metrics=dict(o.metrics)) for o in self.observations]}
def _summary(rows:Sequence[ExperimentObservation])->dict[str,float]:
    if not rows:return {"n":0.0}
    keys={k for r in rows for k in r.metrics}; return {"n":float(len(rows)),**{k:mean(float(r.metrics.get(k,0)) for r in rows) for k in keys}}
def _runtime_metrics(runtime:Any)->Mapping[str,float]: return {k:float(v) for k,v in runtime.cognitive_metrics().items()}
class ConsciousnessResearchHarness:
    def run_cell(self,config:ExperimentConfig,observations:Iterable[Any])->ExperimentResult:
        sub=CognitiveSubstrate(switches=config.switches); state=CognitiveState(switches=config.switches); active=ActiveCognition() if config.embodied else None; active_state=ActiveCognitionState(); rows=[]
        for step,observation in enumerate(observations):
            if step>=config.steps:break
            state=sub.transition(state,observation,confidence=1.0); action=None; metrics={"workspace":float(bool(state.workspace.active)),"ignition":float(bool(state.workspace.last_ignition)),"broadcast_coverage":state.metrics.broadcast_coverage,"prediction_error":state.metrics.prediction_error,"uncertainty":state.prediction.uncertainty,"integration":state.integration.integration,"recurrence":state.metrics.recurrence,"self_model":state.metrics.self_model_accuracy,"metacognition":state.self_model.metacognitive_signal,"continuity":state.metrics.cognitive_continuity}
            if active:
                active_state=active.transition(active_state,uncertainty=state.prediction.uncertainty,candidate_actions=("observe","reflect","respond"),information_gain={"observe":state.prediction.uncertainty,"reflect":.5*state.prediction.uncertainty,"respond":.1},expected_utility={"respond":.5,"reflect":.2,"observe":.1},risk={"respond":.1},observation=observation); action=active_state.selected_action; metrics.update({"active_information_gain":active_state.information_gain,"homeostatic_stability":active_state.homeostasis.stability,"identity_continuity":active_state.identity.continuity})
            rows.append(ExperimentObservation(step,observation,action,metrics))
        return ExperimentResult(config,tuple(rows),_summary(rows))
    def run_runtime(self,runtime:Any,config:ExperimentConfig,observations:Iterable[Any])->ExperimentResult:
        rows=[]
        for step,observation in enumerate(observations):
            if step>=config.steps:break
            runtime.observe(observation); metrics=dict(_runtime_metrics(runtime)); action=runtime.select_action(("observe","reflect","respond")) if config.embodied else None; metrics=dict(_runtime_metrics(runtime)); rows.append(ExperimentObservation(step,observation,action,metrics))
        return ExperimentResult(config,tuple(rows),_summary(rows))
    @staticmethod
    def ablation_matrix(base:ExperimentConfig)->tuple[ExperimentConfig,...]:
        s=base.switches; out=[base]
        for name in ("workspace","recurrence","predictive_processing","self_model","integration"):
            v={"workspace":s.workspace,"recurrence":s.recurrence,"predictive_processing":s.predictive_processing,"self_model":s.self_model,"integration":s.integration}; v[name]=False; out.append(ExperimentConfig(**{**base.__dict__,"switches":CognitiveSwitches(**v)}))
        return tuple(out)
def paired_effect(baseline:Sequence[float],treatment:Sequence[float])->dict[str,float]:
    if len(baseline)!=len(treatment) or not baseline:raise ValueError("baseline and treatment must have equal non-zero length")
    d=[float(b)-float(a) for a,b in zip(baseline,treatment)]; delta=mean(d); sd=stdev(d) if len(d)>1 else 0.; return {"n":float(len(d)),"mean_difference":delta,"std_difference":sd,"standard_error":sd/sqrt(len(d)),"effect_size":delta/sd if sd else 0.}
def continuity_probe(runtime:Any,interruption_steps:int=1)->dict[str,Any]:
    before=runtime.snapshot(); stopped=runtime.continuous_stop()
    for _ in range(max(0,interruption_steps)):runtime.continuous_tick()
    after=runtime.snapshot(); return {"stopped":stopped,"state_preserved":before["state"]==after["state"],"cognitive_cycle_before":(before.get("cognitive_state") or {}).get("cycle",0),"cognitive_cycle_after":(after.get("cognitive_state") or {}).get("cycle",0)}
def reproducibility_record(result:ExperimentResult)->dict[str,Any]: return {"seed":result.config.seed,"model":result.config.model_name,"environment":result.config.environment_name,"theory":result.config.theory,"mechanisms":THEORY_MECHANISMS.get(result.config.theory,()),"summary":dict(result.summary),"trajectory":[dict(step=o.step,action=o.action,metrics=dict(o.metrics)) for o in result.observations]}
__all__=["ConsciousnessResearchHarness","ExperimentConfig","ExperimentObservation","ExperimentResult","THEORY_MECHANISMS","continuity_probe","paired_effect","reproducibility_record"]
