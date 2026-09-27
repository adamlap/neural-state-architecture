"""Convert resident checkpoint tensors into backend compute representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .quantization import QuantizationRegistry, QuantizedMaterializer, QuantizedTensor
from .safetensors_store import SafetensorsRegionStore


@dataclass
class MaterializationMetrics:
    tensors: int = 0
    quantized_tensors: int = 0
    storage_bytes: int = 0
    materialized_bytes: int = 0


class RegionMaterializer:
    """Materialize only the tensors required by a resident logical region."""

    def __init__(
        self,
        source: SafetensorsRegionStore,
        registry: QuantizationRegistry | None = None,
        materializer: QuantizedMaterializer | None = None,
    ):
        self.source = source
        self.registry = registry or QuantizationRegistry()
        self.materializer = materializer
        self.metrics = MaterializationMetrics()

    def load(self, region: str, device: Any = None) -> dict[str, Any]:
        tensors = self.source.tensors_for_region(region)
        raw = self.source.load_region(region)
        result: dict[str, Any] = {}

        for metadata in tensors:
            value = raw[metadata.name]
            self.metrics.tensors += 1
            self.metrics.storage_bytes += metadata.parameter_bytes
            spec = self.registry.spec_for(metadata.name)
            if spec is None:
                result[metadata.name] = value
                try:
                    self.metrics.materialized_bytes += int(value.numel() * value.element_size())
                except AttributeError:
                    self.metrics.materialized_bytes += metadata.parameter_bytes
                continue

            self.metrics.quantized_tensors += 1
            tensor = QuantizedTensor(
                metadata.name,
                metadata.shape,
                metadata.parameter_bytes,
                spec,
                value,
            )
            if self.materializer is None:
                raise RuntimeError(
                    f"quantized tensor {metadata.name!r} requires a materializer"
                )
            result[metadata.name] = self.materializer.materialize(tensor, device)
            try:
                self.metrics.materialized_bytes += int(
                    result[metadata.name].numel() * result[metadata.name].element_size()
                )
            except AttributeError:
                self.metrics.materialized_bytes += tensor.compute_bytes

        return result
