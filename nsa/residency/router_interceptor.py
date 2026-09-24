"""Runtime interceptor for routing and intra-layer phase transitions.

Supports:
1. MoE models: router hook intercepting gating logits to predict top-k experts
   before the MLP block executes.
2. Dense models: intra-layer transition hooks (attention -> MLP -> next layer)
   to enable sub-layer prefetch pipelining.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from time import monotonic
from typing import Any, Callable, List, Optional, Sequence, Tuple
import torch

from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.types import MemoryTier, ResidencyEvent


@dataclass(frozen=True)
class RoutingPrediction:
    layer_index: int
    selected_regions: Tuple[str, ...]
    probabilities: Tuple[float, ...]
    timestamp: float


RouteCallback = Callable[[RoutingPrediction], None]


class MoERouterHook:
    """Hooks into an MoE gating/router module to intercept top-k routing decisions."""

    def __init__(
        self,
        layer_index: int,
        gate_module: torch.nn.Module,
        num_experts_per_tok: int = 2,
        on_route: Optional[RouteCallback] = None,
        is_block_sparse: bool = False,
    ) -> None:
        self.layer_index = layer_index
        self.gate_module = gate_module
        self.num_experts_per_tok = num_experts_per_tok
        self.on_route = on_route
        self.is_block_sparse = is_block_sparse
        self._hook_handle = None
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return

        def hook_fn(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            # output of gate is usually router_logits or a tuple (logits, ...)
            logits = output[0] if isinstance(output, tuple) else output
            if not isinstance(logits, torch.Tensor):
                return

            with torch.no_grad():
                # Routers commonly emit [batch, seq, experts], but some
                # implementations collapse batch/sequence. Flatten all
                # leading dimensions and consistently inspect the final token.
                if logits.dim() < 2:
                    return
                last_token_logits = logits.reshape(-1, logits.shape[-1])[-1]
                scores = torch.softmax(last_token_logits, dim=-1)
                top_k_scores, top_k_indices = torch.topk(scores, k=min(self.num_experts_per_tok, scores.shape[-1]))

                selected_regions = tuple(
                    f"layer.{self.layer_index}.expert.{idx.item()}"
                    for idx in top_k_indices
                )
                probabilities = tuple(round(p.item(), 4) for p in top_k_scores)

                prediction = RoutingPrediction(
                    layer_index=self.layer_index,
                    selected_regions=selected_regions,
                    probabilities=probabilities,
                    timestamp=monotonic(),
                )

                if self.on_route is not None:
                    self.on_route(prediction)

        self._hook_handle = self.gate_module.register_forward_hook(hook_fn)
        self._installed = True

    def remove(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._installed = False


def instrument_dense_sublayers(
    model: Any,
    manager: NeuralResidencyManager,
    on_sublayer: Optional[Callable[[str], None]] = None,
) -> int:
    """Instrument attention and MLP sub-layers of dense transformer models.

    Allows dense models to pipeline memory materialization between attention
    and MLP computation, giving 2x the time window for overlapping I/O with compute.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0

    count = 0
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)

        if attn is not None and not getattr(attn, "_nsa_sublayer_wrapped", False):
            orig_attn_fwd = attn.forward
            attn_rid = f"layer.{idx}.attn"

            @wraps(orig_attn_fwd)
            def wrapped_attn(*args, __orig=orig_attn_fwd, __rid=attn_rid, __next=f"layer.{idx}.mlp", **kwargs):
                started = monotonic()
                previous = manager.current_region
                if hasattr(manager.predictor, "observe_transition"):
                    manager.predictor.observe_transition(previous, __rid)
                manager.current_region = __rid
                if on_sublayer is not None:
                    # Signal that MLP of this layer will be needed next
                    on_sublayer(__next)
                res = __orig(*args, **kwargs)
                manager.record_event(ResidencyEvent(
                    monotonic(), __rid, "execute",
                    manager.tiers.get(__rid, MemoryTier.RAM),
                    manager.tiers.get(__rid, MemoryTier.RAM),
                    0, (monotonic() - started) * 1000, "dense-attn-forward"
                ))
                return res

            attn.forward = wrapped_attn
            attn._nsa_sublayer_wrapped = True
            count += 1

        if mlp is not None and not getattr(mlp, "_nsa_sublayer_wrapped", False):
            orig_mlp_fwd = mlp.forward
            mlp_rid = f"layer.{idx}.mlp"
            next_attn_rid = f"layer.{idx+1}.attn" if idx + 1 < len(layers) else "lm_head"

            @wraps(orig_mlp_fwd)
            def wrapped_mlp(*args, __orig=orig_mlp_fwd, __rid=mlp_rid, __next=next_attn_rid, **kwargs):
                started = monotonic()
                previous = manager.current_region
                if hasattr(manager.predictor, "observe_transition"):
                    manager.predictor.observe_transition(previous, __rid)
                manager.current_region = __rid
                if on_sublayer is not None:
                    # Signal that attention of next layer will be needed next
                    on_sublayer(__next)
                res = __orig(*args, **kwargs)
                manager.record_event(ResidencyEvent(
                    monotonic(), __rid, "execute",
                    manager.tiers.get(__rid, MemoryTier.RAM),
                    manager.tiers.get(__rid, MemoryTier.RAM),
                    0, (monotonic() - started) * 1000, "dense-mlp-forward"
                ))
                return res

            mlp.forward = wrapped_mlp
            mlp._nsa_sublayer_wrapped = True
            count += 1

    return count