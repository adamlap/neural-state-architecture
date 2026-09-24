"""Universal Governed Neural Substrate Backend for MoE and Dense Models.

Dynamically materializes only active specialists (for MoE architectures) or
pipelined sublayers (for Dense architectures), governed by NSA cognitive state
and predictive routing.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Union
import torch

from nsa.residency import (
    ActiveResidencyController,
    MemoryTier,
    NeuralRegion,
    NeuralResidencyManager,
    ResidencyPolicy,
    ResidencyTrace,
    cognitive_state_features,
)
from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher
from nsa.residency.io_pipeline import AsyncMaterializationPipeline
from nsa.residency.moe_regions import build_moe_regions, detect_moe_spec
from nsa.residency.quantized_storage import QuantizedRegionStore
from nsa.residency.router_interceptor import MoERouterHook, RoutingPrediction, instrument_dense_sublayers
from nsa.residency.sizing import layer_sizes
from nsa.runtime.inference.action_parser import ActionParser
from nsa.runtime.inference.base import BackendMode, InferenceBackend, LLMGenerationOutput
from nsa.core.substrate_coordinator import NeuralSubstrateCoordinator


class SubstrateTransformersBackend(InferenceBackend):
    """Governed Neural Substrate supporting both MoE specialists and Dense sublayers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        model_path: Optional[str] = None,
        mode: Union[BackendMode, str] = BackendMode.CACHED,
        device: str = "cuda",
        vram_budget_gb: float = 4.0,
        ram_budget_gb: float = 8.0,
        prefetch: bool = True,
        quantized_cold_tier: bool = False,
        precision: str = "int8",
        hot_layers: int = 2,
        warm_layers: int = 2,
        lookahead: int = 2,
        dtype: str = "auto",
        offload_folder: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.mode = BackendMode(mode) if isinstance(mode, str) else mode
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.prefetch = prefetch
        self.quantized_cold_tier = quantized_cold_tier
        self.precision = precision
        self.hot_layers = max(0, hot_layers)
        self.warm_layers = max(0, warm_layers)
        self.lookahead = max(1, lookahead)
        self.offload_folder = offload_folder or os.path.join(
            os.path.expanduser("~/.cache/nsa"), "substrate", self.model_name.replace("/", "_").replace(":", "_")
        )

        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.is_moe = False
        self.moe_spec = None

        self._active_residency_state: Mapping[str, object] = {"tags": []}
        self.trace = ResidencyTrace()
        self.residency = NeuralResidencyManager(ResidencyPolicy(
            vram_budget_bytes=int(vram_budget_gb * 1024**3),
            ram_budget_bytes=int(ram_budget_gb * 1024**3),
        ))
        self.residency.trace = self.trace

        self.quantized_store = QuantizedRegionStore(target_precision=precision) if quantized_cold_tier else None
        self.prefetcher: Optional[AccelerateDiskPrefetcher] = None
        self.residency_controller: Optional[ActiveResidencyController] = None
        self.router_hooks: List[MoERouterHook] = []
        self.coordinator = NeuralSubstrateCoordinator(residency=self.residency)

    @staticmethod
    def _resolve_device(device: str) -> str:
        from nsa.runtime.hardware import HardwareDeviceManager
        return HardwareDeviceManager.resolve_device_string(device)

    def load_model(self) -> bool:
        if self._loaded or self.mode == BackendMode.MOCK:
            self._loaded = True
            return True

        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        self.moe_spec = detect_moe_spec(config)
        self.is_moe = self.moe_spec is not None

        # Build regions: MoE granular regions or dense layer regions
        if self.is_moe:
            regions = build_moe_regions(config, self.model_path, spec=self.moe_spec)
        else:
            sizes = layer_sizes(config, self.model_path)
            regions = []
            for i, size in enumerate(sizes):
                half = max(1, int(size // 2))
                regions.extend((
                    NeuralRegion(
                        region_id=f"layer.{i}.attn",
                        parameter_prefixes=(f"model.layers.{i}.self_attn.",),
                        size_bytes=half,
                        layer_index=i,
                        dependencies=(f"layer.{i-1}.mlp",) if i else (),
                    ),
                    NeuralRegion(
                        region_id=f"layer.{i}.mlp",
                        parameter_prefixes=(f"model.layers.{i}.mlp.",),
                        size_bytes=size - half,
                        layer_index=i,
                        dependencies=(f"layer.{i}.attn",),
                    ),
                ))

        self.residency.register(regions)

        # Dispatch via accelerate
        os.makedirs(self.offload_folder, exist_ok=True)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config, trust_remote_code=self.trust_remote_code)
        skeleton.tie_weights()

        # MoE uses fine-grained child mappings so router/shared weights can stay
        # executable while cold experts are independently disk-backed. Dense
        # models retain whole-layer placement for residual safety.
        layer_count = int(getattr(config, "num_hidden_layers", 0))
        device_map = {"model.embed_tokens": self.device, "model.norm": self.device, "lm_head": self.device}
        if self.is_moe:
            for i in range(layer_count):
                hot = i < self.hot_layers
                warm = i < self.hot_layers + self.warm_layers
                shared_tier = self.device if hot else ("cpu" if warm else "cpu")
                expert_tier = self.device if hot else "disk"
                device_map[f"model.layers.{i}.self_attn"] = shared_tier
                device_map[f"model.layers.{i}.input_layernorm"] = shared_tier
                device_map[f"model.layers.{i}.post_attention_layernorm"] = shared_tier
                container = "block_sparse_moe" if "block_sparse_moe" in (self.moe_spec.expert_pattern if self.moe_spec else "") else "mlp"
                gate = f"model.layers.{i}.{container}.gate"
                device_map[gate] = shared_tier
                expert_container = f"model.layers.{i}.{container}.experts"
                for expert in range(self.moe_spec.num_experts if self.moe_spec else 0):
                    device_map[f"{expert_container}.{expert}"] = expert_tier
        else:
            for i in range(layer_count):
                device_map[f"model.layers.{i}"] = self.device if i < self.hot_layers else "disk"

        model = load_checkpoint_and_dispatch(
            skeleton, checkpoint=self.model_path, device_map=device_map,
            offload_folder=self.offload_folder, offload_state_dict=True,
        )
        model.eval()
        self.model = model
        self._loaded = True

        # Setup prefetch and routing hooks
        if self.prefetch:
            self.prefetcher = AccelerateDiskPrefetcher(
                self.offload_folder,
                {r.region_id: r.parameter_prefixes for r in self.residency.regions.values()},
            )
            self.residency_controller = ActiveResidencyController(
                self.residency,
                load_fn=lambda rid, tier: None,
                lookahead=self.lookahead,
                prefetch_fn=self.prefetcher.prefetch,
                prefetch_eligible=lambda rid: self.prefetcher.covers(rid) and self.prefetcher.needs_warming(rid),
            )

            if self.is_moe:
                self._install_moe_router_hooks()
            else:
                instrument_dense_sublayers(
                    self.model, self.residency,
                    on_sublayer=self._on_dense_sublayer_predicted,
                )

        return True

    def _install_moe_router_hooks(self) -> None:
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None:
            return

        for idx, layer in enumerate(layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None) or getattr(getattr(layer, "block_sparse_moe", None), "gate", None)
            if gate is not None:
                hook = MoERouterHook(
                    layer_index=idx,
                    gate_module=gate,
                    num_experts_per_tok=self.moe_spec.num_experts_per_tok if self.moe_spec else 2,
                    on_route=self._on_moe_route_predicted,
                )
                hook.install()
                self.router_hooks.append(hook)

    def _on_moe_route_predicted(self, pred: RoutingPrediction) -> None:
        """Feed early router predictions directly into prefetch scheduler."""
        self.coordinator.observe_routing(pred.selected_regions, pred.probabilities)
        if self.residency_controller is not None:
            self.residency_controller.prefetch_async(self._active_residency_state)

    def _on_dense_sublayer_predicted(self, next_region_id: str) -> None:
        """Feed sublayer transition into prefetcher."""
        if self.residency_controller is not None:
            self.residency_controller.prefetch_async(self._active_residency_state)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        extract_hidden: bool = False,
        state: Optional[Mapping[str, object]] = None,
    ) -> LLMGenerationOutput:
        if self.mode == BackendMode.MOCK:
            return LLMGenerationOutput(
                text='{"thought":"mock-substrate","action":"probe_service_config","params":{},"confidence":0.92}',
                tokens=[1, 2, 3],
                confidence_estimate=0.92,
                raw_response={"residency": self.residency.snapshot().to_dict(), "is_moe": self.is_moe},
            )

        if not self._loaded:
            self.load_model()

        assert self.model is not None and self.tokenizer is not None
        self._active_residency_state = cognitive_state_features(state)
        token_context = dict(self._active_residency_state)
        token_context["prompt"] = prompt[:512]
        self.coordinator.observe_token(__import__("nsa.core.substrate_coordinator", fromlist=["TokenEnvelope"]).TokenEnvelope(token_id=self.residency.current_region or 0, cognitive_state=token_context))
        transition = self.coordinator.plan(self._active_residency_state, reason="generation-start")
        self.coordinator.apply_resource_allocations(transition.allocations)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0, "return_dict_in_generate": True}
        if temperature > 0:
            kwargs["temperature"] = max(0.01, temperature)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)

        generated = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        self.coordinator.commit_execution([self.residency.current_region] if self.residency.current_region else (), reason="generation-complete")
        return LLMGenerationOutput(
            text=text,
            tokens=generated.tolist(),
            confidence_estimate=0.92,
            raw_response={"residency": self.residency.snapshot().to_dict(), "is_moe": self.is_moe},
        )

    def close(self) -> None:
        for hook in self.router_hooks:
            hook.remove()
        self.router_hooks.clear()
        if self.residency_controller is not None:
            self.residency_controller.shutdown(wait=True)
        if self.prefetcher is not None:
            self.prefetcher.close()
        self.trace.close()

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
        fallback_action: str = "probe_service_config",
    ) -> Dict[str, Any]:
        tools = "\n".join(f"- {t.get('name','')}: {t.get('description','')}" for t in available_tools)
        prompt = f"{system_context}\n\nAvailable tools:\n{tools}\n\nReturn JSON with thought, action, params and confidence.\n{task_instruction}"
        out = self.generate(prompt, max_tokens=128, temperature=0.0)
        parsed = ActionParser.extract_action_json(out.text)
        return ActionParser.sanitize_action_proposal(parsed, available_tools, default_fallback=fallback_action, strict_live=True)