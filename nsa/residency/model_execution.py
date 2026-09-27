"""Compile model configuration into universal residency and execution graphs."""

from __future__ import annotations

from typing import Any, Mapping

from .execution_graph import ExecutionGraph, ExecutionOp
from .types import NeuralRegion


def _layer_types(config: Mapping[str, Any]) -> list[str]:
    text = config.get("text_config", config)
    explicit = text.get("layer_types")
    if explicit:
        return [str(x) for x in explicit]
    count = int(text.get("num_hidden_layers", 0))
    interval = int(text.get("full_attention_interval", 0))
    if interval > 0:
        return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(count)]
    return ["transformer"] * count


def compile_residency_regions(
    config: Mapping[str, Any],
    *,
    region_size_bytes: Mapping[str, int] | None = None,
) -> tuple[NeuralRegion, ...]:
    """Create logical weight/state regions without loading model weights.

    Exact byte sizes can be supplied by the weight index. When unavailable,
    regions intentionally default to zero rather than fabricating parameter
    sizes from architecture heuristics.
    """
    sizes = region_size_bytes or {}
    text = config.get("text_config", config)
    layer_types = _layer_types(config)
    regions: list[NeuralRegion] = [
        NeuralRegion(
            "embeddings",
            parameter_prefixes=("model.embed_tokens", "model.language_model.embed_tokens"),
            size_bytes=int(sizes.get("embeddings", 0)),
            semantic_tags=("embedding",),
        )
    ]
    for index, layer_type in enumerate(layer_types):
        kind = "linear_attention" if "linear" in layer_type.lower() or "deltanet" in layer_type.lower() else (
            "full_attention" if "full" in layer_type.lower() else "transformer"
        )
        weight_id = f"layer.{index}.weights"
        state_id = f"layer.{index}.state"
        regions.append(
            NeuralRegion(
                weight_id,
                parameter_prefixes=(f"model.layers.{index}", f"model.language_model.layers.{index}"),
                size_bytes=int(sizes.get(weight_id, 0)),
                layer_index=index,
                semantic_tags=(kind, "weights"),
                dependencies=((f"layer.{index - 1}.weights",) if index else ()),
            )
        )
        if kind == "linear_attention":
            regions.append(
                NeuralRegion(
                    state_id,
                    size_bytes=int(sizes.get(state_id, 0)),
                    layer_index=index,
                    semantic_tags=("linear_attention", "persistent_state"),
                    dependencies=(weight_id,),
                )
            )
        elif kind == "full_attention":
            regions.append(
                NeuralRegion(
                    state_id,
                    size_bytes=int(sizes.get(state_id, 0)),
                    layer_index=index,
                    semantic_tags=("full_attention", "kv_state"),
                    dependencies=(weight_id,),
                )
            )
    regions.append(
        NeuralRegion(
            "lm_head",
            parameter_prefixes=("lm_head", "model.lm_head"),
            size_bytes=int(sizes.get("lm_head", 0)),
            semantic_tags=("output",),
            dependencies=((f"layer.{len(layer_types) - 1}.weights",) if layer_types else ()),
        )
    )
    if "vision_config" in config:
        regions.append(
            NeuralRegion(
                "vision",
                parameter_prefixes=("visual", "model.visual"),
                size_bytes=int(sizes.get("vision", 0)),
                semantic_tags=("vision",),
            )
        )
    if int(text.get("mtp_num_hidden_layers", 0)):
        regions.append(
            NeuralRegion(
                "mtp",
                parameter_prefixes=("mtp", "model.mtp"),
                size_bytes=int(sizes.get("mtp", 0)),
                semantic_tags=("mtp",),
                dependencies=(("lm_head",) if "lm_head" in {r.region_id for r in regions} else ()),
            )
        )
    return tuple(regions)


def compile_execution_graph(config: Mapping[str, Any]) -> ExecutionGraph:
    """Compile one decoder-token execution path from model configuration."""
    layer_types = _layer_types(config)
    graph = ExecutionGraph()
    previous = None

    graph.add(ExecutionOp(
        "embed",
        "embedding",
        output="hidden",
        required_regions=("embeddings",),
        metadata={"architecture": config.get("model_type", "")},
    ))
    previous = "embed"

    for index, layer_type in enumerate(layer_types):
        weight = f"layer.{index}.weights"
        state = f"layer.{index}.state"
        op_id = f"layer.{index}"
        graph.add(
            ExecutionOp(
                op_id,
                "linear_attention" if "linear" in layer_type.lower() or "deltanet" in layer_type.lower() else "attention",
                inputs=("hidden",),
                output="hidden",
                required_regions=(weight, state),
                persistent_regions=(state,),
                depends_on=(previous,),
                metadata={"layer_index": index, "layer_type": layer_type},
            )
        )
        previous = op_id

    graph.add(
        ExecutionOp(
            "lm_head",
            "lm_head",
            inputs=("hidden",),
            output="logits",
            required_regions=("lm_head",),
            depends_on=((previous,) if previous else ("embed",)),
        )
    )
    return graph
