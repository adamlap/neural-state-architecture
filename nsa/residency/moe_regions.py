"""Mixture-of-Experts (MoE) neural region decomposition and sizing.

Decomposes MoE transformer models into shared foundational regions
(embeddings, attention, norms, routers) and fine-grained specialist regions
(individual experts per layer).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nsa.residency.accelerate_prefetch import read_safetensors_header
from nsa.residency.types import NeuralRegion


@dataclass(frozen=True)
class MoEArchitectureSpec:
    """Specification of an MoE model architecture parameter layout."""
    num_experts: int
    num_experts_per_tok: int
    expert_pattern: str  # regex pattern with named groups (?P<layer>\d+) and (?P<expert>\d+)
    shared_layer_pattern: str = r"^model\.layers\.(?P<layer>\d+)\.(?!.*experts\.\d+)"
    has_shared_expert: bool = False
    expert_container: str = "mlp.experts"
    router_path: str = "mlp.gate"


# Known MoE architecture layouts
KNOWN_MOE_PATTERNS = [
    re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."),
    re.compile(r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."),
    re.compile(r"^model\.layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)\."),
]


def detect_moe_spec(config: Any) -> Optional[MoEArchitectureSpec]:
    """Detect MoE architecture parameters from a model configuration."""
    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None) or getattr(config, "n_routed_experts", None)
    if not num_experts or int(num_experts) <= 1:
        return None
    num_experts = int(num_experts)

    num_experts_per_tok = getattr(config, "num_experts_per_tok", None) or getattr(config, "num_selected_experts", None) or 2
    num_experts_per_tok = int(num_experts_per_tok)

    has_shared = bool(getattr(config, "has_shared_expert", False) or getattr(config, "n_shared_experts", 0))

    model_type = str(getattr(config, "model_type", "")).lower()
    if "mixtral" in model_type or "olmoe" in model_type:
        pattern = r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    else:
        pattern = r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."

    return MoEArchitectureSpec(
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_pattern=pattern,
        has_shared_expert=has_shared,
        expert_container="block_sparse_moe.experts" if "block_sparse_moe" in pattern else "mlp.experts",
        router_path="block_sparse_moe.gate" if "block_sparse_moe" in pattern else "mlp.gate",
    )


def infer_moe_spec_from_model(config: Any, model: Any) -> Optional[MoEArchitectureSpec]:
    """Derive MoE module paths from the instantiated model skeleton.

    Configurations identify expert counts, but the module tree is authoritative
    for router/expert placement. This keeps residency compatible with new
    Transformers MoE architectures without model-name conditionals.
    """
    base = detect_moe_spec(config)
    if base is None:
        return None
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return base

    for layer in layers:
        for name, module in layer.named_modules():
            if not name.lower().endswith("experts"):
                continue
            indexed = [child for child, _ in module.named_children() if str(child).isdigit()]
            if len(indexed) < 2:
                continue
            container = name
            parent = container.rsplit(".experts", 1)[0]
            router_path = base.router_path
            owner = getattr(layer, parent.split(".")[0], None)
            for candidate in ("gate", "router"):
                if owner is not None and hasattr(owner, candidate):
                    router_path = parent.split(".")[0] + "." + candidate
                    break
            pattern = (
                r"^model\.layers\.(?P<layer>\d+)\."
                + re.escape(container)
                + r"\.(?P<expert>\d+)\."
            )
            return MoEArchitectureSpec(
                num_experts=base.num_experts,
                num_experts_per_tok=base.num_experts_per_tok,
                expert_pattern=pattern,
                has_shared_expert=base.has_shared_expert,
                expert_container=container,
                router_path=router_path,
            )
    return base


def moe_expert_bytes_from_checkpoint(
    checkpoint_dir: str | Path,
    spec: Optional[MoEArchitectureSpec] = None,
) -> Dict[Tuple[int, int], int]:
    """Calculate exact byte sizes for each (layer_idx, expert_idx) from safetensors."""
    root = Path(checkpoint_dir).expanduser()
    if not root.is_dir():
        return {}

    index_file = root / "model.safetensors.index.json"
    file_of: dict[str, str] = {}
    if index_file.is_file():
        try:
            weight_map = json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {})
            file_of = {str(k): str(v) for k, v in weight_map.items()}
        except (OSError, ValueError):
            file_of = {}

    files = sorted(set(file_of.values())) or [p.name for p in sorted(root.glob("*.safetensors"))]
    if not files:
        return {}

    regex = re.compile(spec.expert_pattern) if spec else None
    compiled_patterns = [regex] if regex else KNOWN_MOE_PATTERNS

    sizes: Dict[Tuple[int, int], int] = {}
    for name in files:
        try:
            header = read_safetensors_header(root / name)
        except (OSError, ValueError):
            continue

        for key, (start, end) in header.items():
            if file_of and file_of.get(key) != name:
                continue
            for pat in compiled_patterns:
                match = pat.search(key)
                if match:
                    layer = int(match.group("layer"))
                    expert = int(match.group("expert"))
                    sizes[(layer, expert)] = sizes.get((layer, expert), 0) + (end - start)
                    break

    return sizes


def estimate_moe_expert_bytes(config: Any, bytes_per_param: int = 2) -> int:
    """Estimate the parameter bytes of a single expert MLP in an MoE layer."""
    hidden = int(getattr(config, "hidden_size", 0))
    inter = getattr(config, "moe_intermediate_size", None) or getattr(config, "intermediate_size", hidden * 4)
    inter = int(inter)
    params = 3 * hidden * inter
    return int(params * bytes_per_param)


def estimate_moe_shared_bytes(config: Any, bytes_per_param: int = 2) -> int:
    """Estimate the parameter bytes of the shared components of an MoE layer."""
    hidden = int(getattr(config, "hidden_size", 0))
    heads = max(1, int(getattr(config, "num_attention_heads", 1)))
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    kv_dim = kv_heads * head_dim
    q_dim = heads * head_dim

    attention = hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
    norms = 2 * hidden
    num_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", 8)
    router = hidden * int(num_experts)

    shared_expert_bytes = 0
    if getattr(config, "has_shared_expert", False) or getattr(config, "n_shared_experts", 0):
        shared_inter = int(getattr(config, "shared_expert_intermediate_size", hidden * 4))
        shared_expert_bytes = 3 * hidden * shared_inter

    total_params = attention + norms + router + shared_expert_bytes
    return int(total_params * bytes_per_param)


def build_moe_regions(
    config: Any,
    checkpoint_dir: Optional[str | Path] = None,
    bytes_per_param: int = 2,
    spec: Optional[MoEArchitectureSpec] = None,
) -> List[NeuralRegion]:
    """Construct granular NeuralRegions for an MoE architecture."""
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    moe_spec = spec or detect_moe_spec(config)
    num_experts = moe_spec.num_experts if moe_spec else int(getattr(config, "num_local_experts", 8))

    expert_sizes = moe_expert_bytes_from_checkpoint(checkpoint_dir, moe_spec) if checkpoint_dir else {}
    est_expert_size = estimate_moe_expert_bytes(config, bytes_per_param)
    est_shared_size = estimate_moe_shared_bytes(config, bytes_per_param)

    regions: List[NeuralRegion] = []
    expert_container = moe_spec.expert_container if moe_spec else "mlp.experts"
    parent_container = expert_container.rsplit(".experts", 1)[0]
    router_path = moe_spec.router_path if moe_spec else "mlp.gate"

    for layer_idx in range(num_layers):
        shared_id = f"layer.{layer_idx}.shared"
        shared_prefixes = (
            f"model.layers.{layer_idx}.input_layernorm.",
            f"model.layers.{layer_idx}.post_attention_layernorm.",
            f"model.layers.{layer_idx}.self_attn.",
            f"model.layers.{layer_idx}.{router_path}.",
        )
        if moe_spec and moe_spec.has_shared_expert:
            shared_prefixes = shared_prefixes + (
                f"model.layers.{layer_idx}.mlp.shared_expert.",
                f"model.layers.{layer_idx}.mlp.shared_expert_gate.",
            )

        regions.append(NeuralRegion(
            region_id=shared_id,
            parameter_prefixes=shared_prefixes,
            size_bytes=est_shared_size,
            layer_index=layer_idx,
            semantic_tags=(f"layer-{layer_idx}", "shared-base", "attention", "router"),
            dependencies=(f"layer.{layer_idx-1}.shared",) if layer_idx > 0 else (),
        ))

        for expert_idx in range(num_experts):
            expert_id = f"layer.{layer_idx}.expert.{expert_idx}"
            prefix = f"model.layers.{layer_idx}.{expert_container}.{expert_idx}."

            actual_size = expert_sizes.get((layer_idx, expert_idx), est_expert_size)
            regions.append(NeuralRegion(
                region_id=expert_id,
                parameter_prefixes=(prefix,),
                size_bytes=actual_size,
                layer_index=layer_idx,
                semantic_tags=(f"layer-{layer_idx}", f"expert-{expert_idx}", "moe-specialist"),
                dependencies=(shared_id,),
            ))

    return regions