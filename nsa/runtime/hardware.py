"""Heterogeneous Hardware Acceleration Bridge.

Provides a unified device abstraction across NVIDIA CUDA, AMD Radeon (DirectML / ROCm),
Apple Silicon MPS, and optimized multi-threaded CPU execution.
Allows NSA neural residency, latent fields, and model backends to automatically
detect and exploit the optimal available accelerator on any host.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple
import torch


class HardwareBackend(str, Enum):
    CUDA = "cuda"
    DIRECTML = "directml"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceDescriptor:
    backend: HardwareBackend
    device_name: str
    target_string: str
    total_memory_bytes: int = 0
    supports_pinned_memory: bool = False
    supports_async_transfers: bool = False

    @property
    def is_accelerated(self) -> bool:
        return self.backend != HardwareBackend.CPU

    @property
    def total_memory_gb(self) -> float:
        return round(self.total_memory_bytes / (1024**3), 2)


class HardwareDeviceManager:
    """Detects and manages compute accelerators across vendors."""

    @staticmethod
    def detect_backends() -> Tuple[DeviceDescriptor, ...]:
        devices: list[DeviceDescriptor] = []

        # 1. NVIDIA CUDA / AMD ROCm (via PyTorch CUDA API)
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA Device"
            try:
                mem = torch.cuda.get_device_properties(0).total_memory if hasattr(torch.cuda, "get_device_properties") else 0
            except Exception:
                mem = 0
            is_hip = bool(getattr(torch.version, "hip", None))
            backend = HardwareBackend.ROCM if is_hip else HardwareBackend.CUDA
            devices.append(DeviceDescriptor(
                backend=backend,
                device_name=name,
                target_string="cuda",
                total_memory_bytes=mem,
                supports_pinned_memory=True,
                supports_async_transfers=True,
            ))

        # 2. Microsoft DirectML (AMD Radeon / Intel / Qualcomm on Windows & WSL)
        try:
            import torch_directml
            if torch_directml.is_available():
                count = torch_directml.device_count()
                for i in range(count):
                    dml_name = torch_directml.device_name(i)
                    devices.append(DeviceDescriptor(
                        backend=HardwareBackend.DIRECTML,
                        device_name=dml_name,
                        target_string=str(torch_directml.device(i)),
                        total_memory_bytes=0,
                        supports_pinned_memory=False,
                        supports_async_transfers=True,
                    ))
        except (ImportError, Exception):
            pass

        # 3. Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append(DeviceDescriptor(
                backend=HardwareBackend.MPS,
                device_name="Apple Silicon MPS",
                target_string="mps",
                supports_pinned_memory=False,
                supports_async_transfers=False,
            ))

        # 4. Standard CPU fallback
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        devices.append(DeviceDescriptor(
            backend=HardwareBackend.CPU,
            device_name=f"CPU ({cpu_count} cores)",
            target_string="cpu",
            supports_pinned_memory=False,
            supports_async_transfers=False,
        ))

        return tuple(devices)

    @classmethod
    def get_optimal_device(cls, preference: Optional[str] = None) -> DeviceDescriptor:
        """Resolve requested preference or auto-select optimal accelerator."""
        available = cls.detect_backends()
        pref = str(preference or "auto").lower()

        if pref != "auto":
            for d in available:
                if d.backend.value == pref or d.target_string == pref:
                    return d

        # Preference priority: CUDA > DirectML > ROCm > MPS > CPU
        for priority in (HardwareBackend.CUDA, HardwareBackend.DIRECTML, HardwareBackend.ROCM, HardwareBackend.MPS):
            for d in available:
                if d.backend == priority:
                    return d

        # Fallback to CPU
        return next(d for d in available if d.backend == HardwareBackend.CPU)

    @classmethod
    def resolve_device_string(cls, wanted: Optional[str] = None) -> str:
        """Helper returning target device string directly (e.g. 'cuda:0', 'cpu', 'dml')."""
        return cls.get_optimal_device(wanted).target_string

    @classmethod
    def to_device(cls, obj: Any, device: Optional[DeviceDescriptor | str] = None) -> Any:
        target = device.target_string if isinstance(device, DeviceDescriptor) else cls.resolve_device_string(device)
        return obj.to(target)