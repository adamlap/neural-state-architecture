"""Hardware-agnostic model topology and residency planning primitives.

This module deliberately knows nothing about Raspberry Pi, CUDA, ROCm, Metal,
or any particular accelerator.  It turns an instantiated model skeleton into
an ordered region graph and estimates the storage footprint of each region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelRegion:
    """A logical model region that can be independently resident."""

    name: str
    kind: str
    layer_index: int | None = None
    parameter_count: int = 0
    parameter_bytes: int = 0
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidencyPlan:
    """Ordered regions plus a bounded working-set budget."""

    regions: tuple[ModelRegion, ...]
    working_set_bytes: int
    storage_bytes: int

    @property
    def region_count(self) -> int:
        return len(self.regions)


def _parameter_bytes(module: Any) -> tuple[int, int]:
    count = 0
    size = 0
    for parameter in module.parameters(recurse=True):
        count += int(parameter.numel())
        size += int(parameter.numel()) * int(parameter.element_size())
    return count, size


def _layer_modules(model: Any) -> Iterable[tuple[int, Any]]:
    layers = None
    for path in (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "transformer.h",
        "transformer.layers",
    ):
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
            if hasattr(current, "__len__") and hasattr(current, "__getitem__"):
                layers = current
                break
        except AttributeError:
            continue

    if layers is None:
        return ()

    return ((i, layers[i]) for i in range(len(layers)))


def profile_model_topology(model: Any) -> ResidencyPlan:
    """Profile a model skeleton without requiring its weights to be resident.

    The function works with ordinary PyTorch modules and uses only structural
    introspection.  It therefore remains portable across CPU, GPU, NPU and
    future backends.
    """

    regions: list[ModelRegion] = []
    storage_bytes = 0

    try:
        embedding = model.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        embedding = None

    if embedding is not None:
        count, size = _parameter_bytes(embedding)
        regions.append(ModelRegion("embeddings", "embedding", parameter_count=count,
                                    parameter_bytes=size))
        storage_bytes += size

    for index, layer in _layer_modules(model):
        count, size = _parameter_bytes(layer)
        kind = "hybrid_layer"
        names = [name.lower() for name, _ in layer.named_modules()]
        if any("deltanet" in name or "linear_attention" in name for name in names):
            kind = "deltanet_layer"
        elif any("attention" in name or "attn" in name for name in names):
            kind = "attention_layer"

        previous = regions[-1].name if regions else ""
        dependencies = (previous,) if previous else ()
        regions.append(ModelRegion(
            name=f"layer.{index}",
            kind=kind,
            layer_index=index,
            parameter_count=count,
            parameter_bytes=size,
            dependencies=dependencies,
        ))
        storage_bytes += size

    try:
        head = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        head = None
    if head is not None:
        count, size = _parameter_bytes(head)
        previous = regions[-1].name if regions else ""
        regions.append(ModelRegion(
            "lm_head", "output_head", parameter_count=count, parameter_bytes=size,
            dependencies=(previous,) if previous else (),
        ))
        storage_bytes += size

    working = max((r.parameter_bytes for r in regions), default=0)
    return ResidencyPlan(tuple(regions), working, storage_bytes)
