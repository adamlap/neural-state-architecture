"""Parallel materialization pipeline for neural virtual memory.

Coordinates multi-stage asynchronous data movement:
1. I/O stage: Storage (NVMe/disk) -> Host RAM page cache or buffer.
2. Dequantization stage: INT4/INT8 -> target float precision.
3. Device promotion: Host RAM -> Target execution device (VRAM/CPU).

Hides storage and dequantization latency behind active compute for both
MoE specialists and Dense layer/sublayer parameter regions.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, Dict, Optional, Set
import torch

from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.quantized_storage import QuantizedRegionStore
from nsa.residency.types import MemoryTier, ResidencyEvent


@dataclass(frozen=True)
class PipelineTask:
    region_id: str
    target_tier: MemoryTier
    precision: str = "float"  # "float", "int8", "int4"
    priority: float = 1.0


class AsyncMaterializationPipeline:
    """Multi-stage worker pipeline for overlapping I/O, decompression, and compute."""

    def __init__(
        self,
        manager: NeuralResidencyManager,
        quantized_store: Optional[QuantizedRegionStore] = None,
        max_io_workers: int = 2,
        max_dequant_workers: int = 2,
        io_reader: Optional[Callable[[str], Optional[int]]] = None,
        device_loader: Optional[Callable[[str, MemoryTier], None]] = None,
    ) -> None:
        self.manager = manager
        self.quantized_store = quantized_store
        self.io_reader = io_reader
        self.device_loader = device_loader

        self._io_executor = ThreadPoolExecutor(max_workers=max(1, max_io_workers), thread_name_prefix="nsa-io")
        self._dequant_executor = ThreadPoolExecutor(max_workers=max(1, max_dequant_workers), thread_name_prefix="nsa-dequant")

        self._lock = Lock()
        self._inflight: Set[str] = set()
        self._futures: Dict[str, Future[None]] = {}
        self._staged_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
        self._closed = False

    def schedule_materialization(self, task: PipelineTask) -> bool:
        """Schedule pipelined materialization of a region if not already resident or inflight."""
        if self._closed:
            return False

        with self._lock:
            if task.region_id in self._inflight:
                return False
            curr_state = self.manager.states.get(task.region_id)
            curr_tier = self.manager.tiers.get(task.region_id)
            if curr_state is not None and curr_state.value == "resident" and curr_tier == task.target_tier:
                return False

            self._inflight.add(task.region_id)
            future = self._io_executor.submit(self._run_pipeline, task)
            self._futures[task.region_id] = future

        future.add_done_callback(lambda _: self._finish(task.region_id))
        return True

    def _run_pipeline(self, task: PipelineTask) -> None:
        start_time = monotonic()
        rid = task.region_id
        region = self.manager.regions.get(rid)
        region_bytes = region.size_bytes if region else 0

        # Stage 1: I/O (Disk -> Host RAM)
        io_start = monotonic()
        self.manager.record_event(ResidencyEvent(
            monotonic(), rid, "pipeline-io-start", MemoryTier.NVME, MemoryTier.RAM,
            0, 0.0, f"priority={task.priority:.2f}"
        ))

        warmed = None
        if self.io_reader is not None:
            try:
                warmed = self.io_reader(rid)
            except Exception as exc:
                self.manager.record_event(ResidencyEvent(
                    monotonic(), rid, "pipeline-io-error", MemoryTier.NVME, MemoryTier.RAM,
                    0, (monotonic() - io_start) * 1000, str(exc)
                ))
                return

        io_latency = (monotonic() - io_start) * 1000
        bytes_read = region_bytes if warmed is None else int(warmed)
        self.manager.record_event(ResidencyEvent(
            monotonic(), rid, "pipeline-io-complete", MemoryTier.NVME, MemoryTier.RAM,
            bytes_read, io_latency, "stage-1-io-done"
        ))

        # Stage 2: Dequantization / Unpacking (if applicable)
        if self.quantized_store is not None and self.quantized_store.contains(rid):
            dequant_start = monotonic()
            dequant_future = self._dequant_executor.submit(self.quantized_store.load_region, rid)
            tensors = dequant_future.result()
            dequant_latency = (monotonic() - dequant_start) * 1000
            if tensors is not None:
                with self._lock:
                    self._staged_buffers[rid] = tensors
                self.manager.record_event(ResidencyEvent(
                    monotonic(), rid, "pipeline-dequant-complete", MemoryTier.RAM, MemoryTier.RAM,
                    region_bytes, dequant_latency, f"precision={task.precision}"
                ))

        # Stage 3: Promotion to target execution tier
        if self.device_loader is not None and task.target_tier != MemoryTier.RAM:
            dev_start = monotonic()
            self.device_loader(rid, task.target_tier)
            dev_latency = (monotonic() - dev_start) * 1000
            self.manager.record_resident(rid, task.target_tier, reason="pipeline-promoted", latency_ms=dev_latency)
        else:
            self.manager.record_resident(rid, task.target_tier, reason="pipeline-resident", latency_ms=(monotonic() - start_time) * 1000)

    def _finish(self, region_id: str) -> None:
        with self._lock:
            self._inflight.discard(region_id)
            self._futures.pop(region_id, None)

    def get_staged_tensors(self, region_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve dequantized tensors that were staged in host memory."""
        with self._lock:
            return self._staged_buffers.get(region_id)

    def evict_staged(self, region_id: str) -> None:
        with self._lock:
            self._staged_buffers.pop(region_id, None)

    def wait(self, region_id: Optional[str] = None, timeout: Optional[float] = None) -> None:
        with self._lock:
            futures = [self._futures.get(region_id)] if region_id else list(self._futures.values())
        for f in futures:
            if f is not None:
                f.result(timeout=timeout)

    def shutdown(self, wait: bool = True) -> None:
        self._closed = True
        self._io_executor.shutdown(wait=wait)
        self._dequant_executor.shutdown(wait=wait)