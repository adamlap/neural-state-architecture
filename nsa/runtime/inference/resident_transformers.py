"""Transformers backend with selective neural storage.

The backend creates an empty model skeleton and lets Hugging Face Accelerate
dispatch checkpoint-backed weights across fast memory and disk. NSA's
residency manager supplies region-level prediction, planning and telemetry.
"""
from __future__ import annotations
import os
from typing import Any, Dict, List, Mapping, Optional, Union
from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import BackendMode, InferenceBackend, LLMGenerationOutput
from nsa.residency import ActiveResidencyController, MemoryTier, NeuralRegion, NeuralResidencyManager, ResidencyPolicy, instrument_decoder_layers, cognitive_state_features, ResidencyTrace
from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher

class SelectiveStorageTransformersBackend(InferenceBackend):
    """Disk-backed Transformers inference with NSA residency planning."""

    def __init__(self, model_name: str="Qwen/Qwen2.5-3B-Instruct", model_path: Optional[str]=None,
                 mode: Union[BackendMode,str]=BackendMode.CACHED, device: str="cuda",
                 vram_budget_gb: float=4.0, ram_budget_gb: float=8.0,
                 prefetch: bool=True, hot_layers: int=2, warm_layers: int=2, no_split_module_classes: Optional[List[str]]=None) -> None:
        self.model_name=model_name
        self.model_path=model_path or model_name
        self.mode=BackendMode(mode) if isinstance(mode,str) else mode
        import torch
        self.device=torch.device(device if device!="auto" and torch.cuda.is_available() else "cpu")
        self.prefetch=prefetch
        self.hot_layers=max(0,hot_layers)
        self.warm_layers=max(0,warm_layers)
        self.no_split_module_classes=no_split_module_classes or ["Qwen2DecoderLayer","Qwen3DecoderLayer"]
        self.model=None
        self.tokenizer=None
        self._loaded=False
        self.trace=ResidencyTrace()
        self.residency=NeuralResidencyManager(ResidencyPolicy(
            vram_budget_bytes=int(vram_budget_gb*1024**3),
            ram_budget_bytes=int(ram_budget_gb*1024**3),
        ))

    def _build_regions(self, config: Any) -> list[NeuralRegion]:
        layers=int(getattr(config,"num_hidden_layers",0))
        hidden=int(getattr(config,"hidden_size",0))
        intermediate=int(getattr(config,"intermediate_size",hidden*4))
        per_layer=int((4*hidden*hidden+4*hidden*intermediate+4*hidden)*2)
        return [NeuralRegion(
            region_id=f"layer.{i}",
            parameter_prefixes=(f"model.layers.{i}.",),
            size_bytes=per_layer, layer_index=i,
            semantic_tags=("transformer",),
            dependencies=(f"layer.{i-1}",) if i else (),
        ) for i in range(layers)]

    def _selective_device_map(self, layer_count: int) -> dict[str, str]:
        """Construct an explicit VRAM/RAM/disk map for decoder regions."""
        device = str(self.device)
        mapping = {"model.embed_tokens": device, "model.norm": device, "lm_head": device}
        hot = set(range(min(self.hot_layers, layer_count)))
        hot.update(range(max(0, layer_count - self.hot_layers), layer_count))
        warm = set(range(self.hot_layers, min(layer_count, self.hot_layers + self.warm_layers)))
        warm.update(range(max(self.hot_layers, layer_count-self.hot_layers-self.warm_layers), max(self.hot_layers, layer_count-self.hot_layers)))
        for i in range(layer_count):
            mapping[f"model.layers.{i}"] = device if i in hot else ("cpu" if i in warm else "disk")
        return mapping
    def load_model(self) -> bool:
        if self._loaded: return True
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Selective storage requires: pip install 'neural-state-architecture[ml-residency]'") from exc
        local_only=self.mode==BackendMode.CACHED
        config=AutoConfig.from_pretrained(self.model_path,local_files_only=local_only)
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_path,local_files_only=local_only,trust_remote_code=True)
        with init_empty_weights():
            model=AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        offload_folder=os.path.join(os.path.expanduser("~/.cache/nsa"),"residency",
                                    self.model_name.replace("/","_").replace(":","_"))
        os.makedirs(offload_folder,exist_ok=True)
        device_map=self._selective_device_map(int(getattr(config,"num_hidden_layers",0)))
        model=load_checkpoint_and_dispatch(
            model,checkpoint=self.model_path,device_map=device_map,
            no_split_module_classes=self.no_split_module_classes,
            offload_folder=offload_folder,offload_state_dict=True,
        )
        model.eval()
        self.model=model
        self._loaded=True
        self.residency.register(self._build_regions(config))
        self.residency.trace = self.trace
        self._active_residency_state: Mapping[str, object] = {"tags": []}
        self.prefetcher = AccelerateDiskPrefetcher(
            offload_folder,
            {region.region_id: region.parameter_prefixes for region in self.residency.regions.values()},
        ) if self.prefetch else None
        self.residency_controller = ActiveResidencyController(
            self.residency,
            load_fn=self._unsupported_physical_prefetch,
            lookahead=2,
            prefetch_fn=(self.prefetcher.prefetch if self.prefetcher is not None else None),
        )
        instrument_decoder_layers(
            self.model,
            self.residency,
            on_region=(self._on_region_execute if self.prefetch else None),
        )
        return True

    @staticmethod
    def _unsupported_physical_prefetch(region_id: str, tier: MemoryTier) -> None:
        raise RuntimeError(
            "Direct physical residency transfer is owned by Accelerate; "
            "use prefetch_async() for the page-cache prefetch path."
        )

    def _on_region_execute(self, region_id: str) -> None:
        if getattr(self, "residency_controller", None) is not None:
            self.residency_controller.prefetch_async(self._active_residency_state)

    def _state_tags(self,prompt:str)->Mapping[str,object]:
        return {"tags":[w.lower() for w in prompt.split() if len(w)>4][:8]}

    def generate(self,prompt:str,max_tokens:int=256,temperature:float=0.7,
                 extract_hidden:bool=False,state:Optional[Mapping[str,object]]=None)->LLMGenerationOutput:
        if self.mode==BackendMode.MOCK:
            return LLMGenerationOutput(text='{"thought":"mock","action":"probe_service_config","params":{},"confidence":0.88}',tokens=[1,2,3],confidence_estimate=0.88)
        if not self._loaded: self.load_model()
        import torch
        assert self.model is not None and self.tokenizer is not None
        self._active_residency_state = cognitive_state_features(state) if state is not None else self._state_tags(prompt)
        decisions=self.residency.plan(self._active_residency_state)
        for decision in decisions: self.residency.scores[decision.region_id]=decision.score
        inputs=self.tokenizer(prompt,return_tensors="pt")
        input_device=next(self.model.parameters()).device
        inputs={k:v.to(input_device) for k,v in inputs.items()}
        kwargs={"max_new_tokens":max_tokens,"do_sample":temperature>0,"return_dict_in_generate":True,"output_hidden_states":extract_hidden}
        if temperature>0: kwargs["temperature"]=max(0.01,temperature)
        with torch.no_grad(): outputs=self.model.generate(**inputs,**kwargs)
        generated=outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text=self.tokenizer.decode(generated,skip_special_tokens=True)
        return LLMGenerationOutput(
            text=text,tokens=generated.tolist(),
            hidden_states=(outputs.hidden_states[-1][-1] if extract_hidden and outputs.hidden_states else None),
            confidence_estimate=0.90,
            raw_response={"residency":self.residency.snapshot().__dict__},
        )

    def close(self) -> None:
        controller = getattr(self, "residency_controller", None)
        if controller is not None:
            controller.shutdown(wait=True)

    def propose_action(self,system_context:str,task_instruction:str,
                       available_tools:List[Dict[str,Any]],fallback_action:str="probe_service_config")->Dict[str,Any]:
        tools="\n".join(f"- {t.get('name','')}: {t.get('description','')}" for t in available_tools)
        prompt=f"{system_context}\n\nAvailable tools:\n{tools}\n\nReturn JSON with thought, action, params and confidence.\n{task_instruction}"
        out=self.generate(prompt,max_tokens=128,temperature=0.0)
        parsed=ActionParser.extract_action_json(out.text)
        return ActionParser.sanitize_action_proposal(parsed,available_tools,default_fallback=fallback_action,strict_live=True)
