from nsa.residency.materialization import RegionMaterializer
from nsa.residency.quantization import QuantizationRegistry, QuantizationSpec, QuantizedMaterializer
from nsa.residency.weight_index import TensorRegion


class FakeTensor:
    def __init__(self, elements, element_size):
        self._elements = elements
        self._element_size = element_size

    def numel(self):
        return self._elements

    def element_size(self):
        return self._element_size


class FakeSource:
    def __init__(self):
        self.tensor = TensorRegion(
            "layer.0.weight", "a", 8, (4, 4), "F4_PACKED", "layer.0"
        )

    def tensors_for_region(self, region):
        return (self.tensor,)

    def load_region(self, region):
        return {"layer.0.weight": FakeTensor(16, 2)}


def test_region_materializer_dispatches_quantized_tensor():
    registry = QuantizationRegistry()
    registry.register(
        "layer.0.weight",
        QuantizationSpec("test", 4, block_shape=(4, 4)),
    )
    seen = []

    def materialize(tensor, device):
        seen.append((tensor.storage_bytes, device))
        return FakeTensor(16, 2)

    region_materializer = RegionMaterializer(
        FakeSource(),
        registry,
        QuantizedMaterializer({"test": materialize}),
    )
    result = region_materializer.load("layer.0", device="cpu")

    assert result["layer.0.weight"].numel() == 16
    assert seen == [(8, "cpu")]
    assert region_materializer.metrics.quantized_tensors == 1
    assert region_materializer.metrics.storage_bytes == 8
    assert region_materializer.metrics.materialized_bytes == 32
