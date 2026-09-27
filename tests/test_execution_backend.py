from nsa.residency.execution_backend import CpuBackend, DeviceInfo, value_nbytes


def test_value_nbytes_is_library_agnostic():
    assert value_nbytes(b"abcd") == 4

    class TensorLike:
        def numel(self):
            return 12
        def element_size(self):
            return 2

    assert value_nbytes(TensorLike()) == 24


def test_cpu_backend_transfer_metrics_are_device_aware():
    backend = CpuBackend()
    other = DeviceInfo("accelerator", "npu", 1024)
    value = b"12345"
    backend.move(value, backend.device_info())
    backend.move(value, other)
    backend.release(value)
    assert backend.metrics.moves == 2
    assert backend.metrics.bytes_moved == 5
    assert backend.metrics.releases == 1
