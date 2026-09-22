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
from nsa.residency.sizing import layer_sizes

class SelectiveStorageTransformersBackend(InferenceBackend):
    """Disk-backed Transformers inference with NSA residency planning."""

    @classmethod
    def from_local_model(cls, key: str, **kwargs: Any) -> "SelectiveStorageTransformersBackend":
        from nsa.runtime.inference.model_registry import get_local_model
        spec = get_local_model(key)
        return cls(
            model_name=spec.model_id,
            model_path=spec.checkpoint_path(),
            vram_budget_gb=spec.vram_budget_gb,
            ram_budget_gb=spec.ram_budget_gb,
            **kwargs,
        )

    def __init__(self, model_name: str="Qwen/Qwen2.5-3B-Instruct", model_path: Optional[str]=None,
                 mode: Union[BackendMode,str]=BackendMode.CACHED, device: str="cuda",
                 vram_budget_gb: float=4.0, ram_budget_gb: float=8.0,
                 prefetch: bool=True, hot_layers: int=2, warm_layers: int=2, no_split_module_classes: Optional[List[str]]=None,
                 dtype: str="auto", offload_folder: Optional[str]=None, trust_remote_code: bool=False) -> None:
        self.model_name=model_name
        self.model_path=model_path or model_name
        self.mode=BackendMode(mode) if isinstance(mode,str) else mode
        self.device=self._resolve_device(device)
        self.dtype=dtype
        self.trust_remote_code=trust_remote_code
        self.prefetch=prefetch
        self.hot_layers=max(0,hot_layers)
        self.warm_layers=max(0,warm_layers)
        self.no_split_module_classes=no_split_module_classes or ["Qwen2DecoderLayer","Qwen3DecoderLayer"]
        self.offload_folder=offload_folder or os.path.join(
            os.path.expanduser("~/.cache/nsa"),"residency",self.model_name.replace("/","_").replace(":","_"))
        self.model=None
        self.tokenizer=None
        self._loaded=False
        self.prefetcher: Optional[AccelerateDiskPrefetcher]=None
        self.residency_controller: Optional[ActiveResidencyController]=None
        self._active_residency_state: Mapping[str, object] = {"tags": []}
        self.trace=ResidencyTrace()
        self.residency=NeuralResidencyManager(ResidencyPolicy(
            vram_budget_bytes=int(vram_budget_gb*1024**3),
            ram_budget_bytes=int(ram_budget_gb*1024**3),
        ))
        self.residency.trace = self.trace

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Return 'cpu' or a 'cuda[:n]' string; CUDA is only used when present."""
        wanted = str(device or "cpu").lower()
        if wanted == "cpu" or not (wanted == "auto" or wanted.startswith("cuda")):
            return "cpu"
        try:
            import torch
            cuda = torch.cuda.is_available()
        except ImportError:
            cuda = False
        if not cuda:
            return "cpu"
        return "cuda" if wanted == "auto" else wanted

    def _accelerate_device(self) -> Union[int, str]:
        """Accelerate device_map spelling of the fast device."""
        if self.device.startswith("cuda"):
            return int(self.device.split(":")[1]) if ":" in self.device else 0
        return "cpu"

    def _resolve_dtype(self, config: Any) -> Any:
        import torch
        if self.dtype not in (None, "auto"):
            return getattr(torch, str(self.dtype))
        declared = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
        if isinstance(declared, str):
            declared = getattr(torch, declared, None)
        if declared is not None:
            return declared
        return torch.bfloat16 if self.device.startswith("cuda") else torch.float32

    def _local_checkpoint(self) -> str:
        """A local checkpoint directory (downloads only when the mode allows it)."""
        if os.path.isdir(self.model_path):
            return self.model_path
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=self.model_path, local_files_only=self.mode==BackendMode.CACHED)

    def _build_regions(self, config: Any, checkpoint_dir: Optional[str]=None, bytes_per_param: int=2) -> list[NeuralRegion]:
        sizes=layer_sizes(config,checkpoint_dir,bytes_per_param)
        return [NeuralRegion(
            region_id=f"layer.{i}",
            parameter_prefixes=(f"model.layers.{i}.",),
            size_bytes=size, layer_index=i,
            dependencies=(f"layer.{i-1}",) if i else (),
        ) for i,size in enumerate(sizes)]

    def _selective_device_map(self, layer_count: int) -> dict[str, Union[int, str]]:
        """Construct an explicit VRAM/RAM/disk map for decoder regions."""
        device = self._accelerate_device()
        mapping: dict[str, Union[int, str]] = {"model.embed_tokens": device, "model.norm": device, "lm_head": device}
        hot = set(range(min(self.hot_layers, layer_count)))
        hot.update(range(max(0, layer_count - self.hot_layers), layer_count))
        warm = set(range(self.hot_layers, min(layer_count, self.hot_layers + self.warm_layers)))
        warm.update(range(max(self.hot_layers, layer_count-self.hot_layers-self.warm_layers), max(self.hot_layers, layer_count-self.hot_layers)))
        for i in range(layer_count):
            mapping[f"model.layers.{i}"] = device if i in hot else ("cpu" if i in warm else "disk")
        return mapping

    def _record_static_placement(self, device_map: Mapping[str, Union[int, str]]) -> None:
        """Record where Accelerate physically placed each region at dispatch time."""
        for rid, region in self.residency.regions.items():
            target = device_map.get(f"model.layers.{region.layer_index}")
            if target == "disk" or target is None:
                continue
            tier = MemoryTier.RAM if target == "cpu" else MemoryTier.VRAM
            self.residency.record_resident(rid, tier, reason="device-map")

    def load_model(self) -> bool:
        if self._loaded: return True
        try:
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Selective storage requires: pip install 'neural-state-architecture[ml-residency]'") from exc
        import torch
        from transformers.utils import is_torch_available
        if not is_torch_available():
            raise RuntimeError(
                f"transformers cannot use the installed torch ({torch.__version__}); "
                "transformers>=5 requires torch>=2.5. Upgrade torch or install transformers<5."
            )
        checkpoint=self._local_checkpoint()
        local_only=self.mode==BackendMode.CACHED
        config=AutoConfig.from_pretrained(checkpoint,local_files_only=local_only,trust_remote_code=self.trust_remote_code)
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=local_only,trust_remote_code=self.trust_remote_code)
        dtype=self._resolve_dtype(config)
        with init_empty_weights():
            model=AutoModelForCausalLM.from_config(config,trust_remote_code=self.trust_remote_code)
            # the skeleton defaults to float32; without this Accelerate upcasts every weight
            model=model.to(dtype)
        # Tied embeddings (Qwen2.5 1.5B/3B) must be tied on the skeleton, otherwise
        # lm_head stays on the meta device and dispatch fails.
        model.tie_weights()
        os.makedirs(self.offload_folder,exist_ok=True)
        layer_count=int(getattr(config,"num_hidden_layers",0))
        device_map=self._selective_device_map(layer_count)
        model=load_checkpoint_and_dispatch(
            model,checkpoint=checkpoint,device_map=device_map,dtype=dtype,
            no_split_module_classes=self.no_split_module_classes,
            offload_folder=self.offload_folder,offload_state_dict=True,
        )
        model.eval()
        self.model=model
        self._loaded=True
        self.residency.register(self._build_regions(config,checkpoint,int(dtype.itemsize)))
        self._record_static_placement(device_map)
        self.prefetcher = AccelerateDiskPrefetcher(
            self.offload_folder,
            {region.region_id: region.parameter_prefixes for region in self.residency.regions.values()},
        ) if self.prefetch else None
        self.residency_controller = ActiveResidencyController(
            self.residency,
            load_fn=self._unsupported_physical_prefetch,
            lookahead=2,
            prefetch_fn=(self.prefetcher.prefetch if self.prefetcher is not None else None),
            prefetch_eligible=(self._prefetch_eligible if self.prefetcher is not None else None),
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
        if self.residency_controller is not None:
            self.residency_controller.prefetch_async(self._active_residency_state)

    def _prefetch_eligible(self, region_id: str) -> bool:
        # covers() is a cheap dict lookup; needs_warming() costs a mincore
        # check but skips scheduling a background task entirely for a region
        # that's already resident, which is the common case once decoding has
        # run for a while (see docs/ACTIVE_RESIDENCY.md).
        return self.prefetcher.covers(region_id) and self.prefetcher.needs_warming(region_id)

    def generate(self,prompt:str,max_tokens:int=256,temperature:float=0.7,
                 extract_hidden:bool=False,state:Optional[Mapping[str,object]]=None)->LLMGenerationOutput:
        if self.mode==BackendMode.MOCK:
            return LLMGenerationOutput(text='{"thought":"mock","action":"probe_service_config","params":{},"confidence":0.88}',tokens=[1,2,3],confidence_estimate=0.88)
        if not self._loaded: self.load_model()
        import torch
        assert self.model is not None and self.tokenizer is not None
        # Residency features come from NSA cognitive state only; prompt text is
        # deliberately not turned into residency tags.
        self._active_residency_state = cognitive_state_features(state)
        decisions=self.residency.plan(self._active_residency_state)
        for decision in decisions: self.residency.scores[decision.region_id]=decision.score
        inputs=self.tokenizer(prompt,return_tensors="pt")
        input_device=next(self.model.parameters()).device
        # causal-LM generate() rejects extras such as token_type_ids on some transformers versions
        inputs={k:v.to(input_device) for k,v in inputs.items() if k in ("input_ids","attention_mask")}
        kwargs={"max_new_tokens":max_tokens,"do_sample":temperature>0,"return_dict_in_generate":True,"output_hidden_states":extract_hidden}
        if temperature>0: kwargs["temperature"]=max(0.01,temperature)
        with torch.no_grad(): outputs=self.model.generate(**inputs,**kwargs)
        generated=outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text=self.tokenizer.decode(generated,skip_special_tokens=True)
        return LLMGenerationOutput(
            text=text,tokens=generated.tolist(),
            hidden_states=(outputs.hidden_states[-1][-1] if extract_hidden and outputs.hidden_states else None),
            confidence_estimate=0.90,
            raw_response={"residency":self.residency.snapshot().to_dict()},
        )

    def close(self) -> None:
        if self.residency_controller is not None:
            self.residency_controller.shutdown(wait=True)
        if self.prefetcher is not None:
            self.prefetcher.close()
        self.trace.close()

    def propose_action(self,system_context:str,task_instruction:str,
                       available_tools:List[Dict[str,Any]],fallback_action:str="probe_service_config")->Dict[str,Any]:
        tools="\n".join(f"- {t.get('name','')}: {t.get('description','')}" for t in available_tools)
        prompt=f"{system_context}\n\nAvailable tools:\n{tools}\n\nReturn JSON with thought, action, params and confidence.\n{task_instruction}"
        out=self.generate(prompt,max_tokens=128,temperature=0.0)
        parsed=ActionParser.extract_action_json(out.text)
        return ActionParser.sanitize_action_proposal(parsed,available_tools,default_fallback=fallback_action,strict_live=True)
