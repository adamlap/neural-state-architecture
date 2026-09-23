"""Tests for heterogeneous hardware acceleration bridge."""
import torch
from nsa.runtime.hardware import DeviceDescriptor, HardwareBackend, HardwareDeviceManager
from nsa.runtime.inference.resident_moe import SubstrateTransformersBackend
from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend


def test_hardware_backend_detection():
    devices = HardwareDeviceManager.detect_backends()
    assert len(devices) >= 1
    # At least CPU is always available
    cpu_device = next((d for d in devices if d.backend == HardwareBackend.CPU), None)
    assert cpu_device is not None
    assert cpu_device.target_string == "cpu"
    assert cpu_device.is_accelerated is False


def test_get_optimal_device_resolution():
    cpu_desc = HardwareDeviceManager.get_optimal_device("cpu")
    assert cpu_desc.backend == HardwareBackend.CPU

    auto_desc = HardwareDeviceManager.get_optimal_device("auto")
    assert isinstance(auto_desc, DeviceDescriptor)
    assert auto_desc.target_string in [d.target_string for d in HardwareDeviceManager.detect_backends()]


def test_to_device_tensor_transfer():
    t = torch.tensor([1.0, 2.0, 3.0])
    transferred = HardwareDeviceManager.to_device(t, "cpu")
    assert transferred.device.type == "cpu"
    assert torch.equal(t, transferred)


def test_backend_device_resolution_integration():
    res_dev = SelectiveStorageTransformersBackend._resolve_device("cpu")
    assert res_dev == "cpu"

    moe_dev = SubstrateTransformersBackend._resolve_device("cpu")
    assert moe_dev == "cpu"