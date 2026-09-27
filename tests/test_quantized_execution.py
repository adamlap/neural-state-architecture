import torch

from nsa.residency.execution_backend import CpuBackend
from nsa.residency.quantization import (
    QuantizationSpec,
    QuantizedMaterializer,
    QuantizedTensor,
    packed_storage_bytes,
)


def test_packed_storage_bytes_rounds_up_bits():
    assert packed_storage_bytes((100,), 4) == 50
    assert packed_storage_bytes((7,), 3) == 3


def test_materializer_dispatches_by_scheme():
    tensor = QuantizedTensor(
        "x", (4,), 2, QuantizationSpec("int4", 4), b"xx"
    )
    materializer = QuantizedMaterializer({
        "int4": lambda value, device: (value.name, device),
    })

    assert materializer.materialize(tensor, "cpu") == ("x", "cpu")


def test_cpu_backend_is_reference_execution_path():
    backend = CpuBackend()
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0], [4.0]])

    result = backend.execute("matmul", (a, b))

    assert result.item() == 11.0
