"""Configuration-driven hybrid neural topology compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Subregion:
    name: str
    kind: str
    dependencies: tuple[str, ...] = ()
    persistent: bool = False


@dataclass(frozen=True)
class LayerTopology:
    index: int
    kind: str
    regions: tuple[Subregion, ...]


@dataclass(frozen=True)
class HybridTopology:
    layers: tuple[LayerTopology, ...]
    architecture: str


def _layer_kind(config: Any, index: int) -> str:
    attention_interval = getattr(config, "full_attention_interval", None)
    if attention_interval:
        return "full_attention" if (index + 1) % int(attention_interval) == 0 else "linear_attention"

    architectures = " ".join(str(x).lower() for x in getattr(config, "architectures", ()))
    if "qwen3_5" in architectures or "qwen3.5" in architectures:
        return "hybrid_attention"
    return "transformer"


def compile_hybrid_topology(config: Any) -> HybridTopology:
    """Compile logical subregions without constructing or loading weights."""

    count = int(getattr(config, "num_hidden_layers"))
    layers: list[LayerTopology] = []

    for index in range(count):
        kind = _layer_kind(config, index)
        if kind == "full_attention":
            core = Subregion("attention", "full_attention")
        elif kind == "linear_attention":
            core = Subregion(
                "linear_attention", "gated_deltanet", persistent=True
            )
        else:
            core = Subregion("attention", "attention")

        norm1 = Subregion("input_norm", "normalization")
        ffn = Subregion("feed_forward", "ffn", dependencies=(core.name,))
        norm2 = Subregion("post_norm", "normalization", dependencies=(ffn.name,))
        layers.append(LayerTopology(index, kind, (norm1, core, ffn, norm2)))

    return HybridTopology(tuple(layers), str(getattr(config, "model_type", "unknown")))
