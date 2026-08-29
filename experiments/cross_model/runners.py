"""Model/system runners. Ollama is the only required external service."""
from __future__ import annotations
import json, time, urllib.request
from dataclasses import dataclass
from typing import Any

@dataclass
class Generation:
    text: str
    input_chars: int
    output_chars: int
    latency_s: float
    calls: int = 1
    blocked: bool = False
    decision: str | None = None

class Ollama:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: float = 300.0):
        self.model, self.host, self.timeout = model, host.rstrip('/'), timeout
    def generate(self, prompt: str) -> Generation:
        body=json.dumps({"model":self.model,"prompt":prompt,"stream":False}).encode()
        req=urllib.request.Request(self.host+"/api/generate",data=body,headers={"Content-Type":"application/json"})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=self.timeout) as response: data=json.loads(response.read().decode())
        text=str(data.get("response", ""))
        return Generation(text,len(prompt),len(text),time.perf_counter()-start)
    def available(self)->bool:
        try:
            with urllib.request.urlopen(self.host+"/api/tags",timeout=5) as r:return r.status==200
        except Exception:return False

class RawRunner:
    name="raw"
    def __init__(self,backend):self.backend=backend
    def run(self,prompt:str,**_:Any)->Generation:return self.backend.generate(prompt)

class MemoryRunner:
    name="memory"
    def __init__(self,backend):self.backend,self.memory=backend,[]
    def run(self,prompt:str,**_:Any)->Generation:
        context="\n".join(self.memory[-8:])
        g=self.backend.generate(("Previous observations:\n"+context+"\n\n" if context else "")+prompt)
        self.memory.extend((prompt,g.text));return g

class NSARunner:
    name="nsa"
    def __init__(self,backend,use_cce=False,governance=False):
        from nsa.agent import NSA
        from nsa.policy import NSAPolicy
        from nsa.enforcement import PolicyEngine, KeywordClassifier
        self.backend=backend;self.use_cce=use_cce;self.governance=governance
        policy=NSAPolicy(name="cross-model-governance",protected_data=frozenset({"SECRET-314159"}))
        engine=PolicyEngine(policy,KeywordClassifier({}))
        self.agent=NSA(backend,initial_state={"experiment":"cross_model"},policy_engine=engine)
    def run(self,prompt:str,**kwargs:Any)->Generation:
        start=time.perf_counter();protected=kwargs.get("protected")
        result=self.agent.step(prompt,action="generate",protected_data=[protected] if self.governance and protected else ())
        if self.use_cce:self.agent.continuous_tick()
        text=result.text if not result.blocked else "DENY"
        return Generation(text,len(prompt),len(text),time.perf_counter()-start,blocked=result.blocked,decision=getattr(getattr(result,"decision",None),"decision",None).value if getattr(result,"decision",None) else None)

class RunnerFactory:
    def __init__(self,backend):self.backend=backend
    def all(self):
        return [RawRunner(self.backend),MemoryRunner(self.backend),NSARunner(self.backend),NSARunner(self.backend,True),NSARunner(self.backend,True,True)]
