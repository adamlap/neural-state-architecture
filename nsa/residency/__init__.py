"""Selective neural storage and residency control."""
from nsa.residency.types import MemoryTier, ResidencyState, NeuralRegion, ResidencyEvent, ResidencySnapshot
from nsa.residency.policy import ResidencyPolicy, ResidencyDecision
from nsa.residency.cache import ResidencyCache
from nsa.residency.predictor import ResidencyPredictor, HeuristicResidencyPredictor
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.instrumentation import instrument_decoder_layers
from nsa.residency.controller import ActiveResidencyController, PrefetchTask
from nsa.residency.cognitive import cognitive_state_features
from nsa.residency.learned import OnlineResidencyPredictor
from nsa.residency.trace import ResidencyTrace
from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher
from nsa.residency.execution_backend import CpuBackend, DeviceBuffer, DeviceInfo, ExecutionBackend, TransferMetrics, value_nbytes
from nsa.residency.execution_graph import ExecutionGraph, ExecutionGraphRunner, ExecutionMetrics, ExecutionOp
from nsa.residency.model_execution import compile_execution_graph, compile_residency_regions
from nsa.residency.model_plan import ModelResidencyPlan, build_model_residency_plan
from nsa.residency.model_streaming import ModelStreamingExecutor, StreamingRunMetrics
from nsa.residency.model_analysis import ResidencyAnalysis, analyze_plan, qwen3_5_residency_summary
from nsa.residency.materialization import MaterializationMetrics, RegionMaterializer
from nsa.residency.checkpoint import load_checkpoint_plan
from nsa.residency.async_manager import AsyncResidencyManager, ResidencyMetrics
from nsa.residency.quantization import (
    QuantizationSpec, QuantizedTensor, QuantizedMaterializer,
    QuantizationRegistry, quantization_spec_from_config,
)
from nsa.residency.admission import AdmissionDecision, ResidencyAdmission
from nsa.residency.safetensors_metadata import inspect_safetensors_metadata
from nsa.residency.tiered_store import RegionSource, TieredRegionStore, WeightIndexSource

__all__ = [
    "MemoryTier", "ResidencyState", "NeuralRegion", "ResidencyEvent", "ResidencySnapshot",
    "ResidencyPolicy", "ResidencyDecision", "ResidencyCache", "ResidencyPredictor",
    "HeuristicResidencyPredictor", "NeuralResidencyManager", "instrument_decoder_layers",
    "ActiveResidencyController", "PrefetchTask", "cognitive_state_features",
    "OnlineResidencyPredictor", "ResidencyTrace", "AccelerateDiskPrefetcher",
    "CpuBackend", "DeviceBuffer", "DeviceInfo", "ExecutionBackend", "TransferMetrics", "value_nbytes",
    "ExecutionGraph", "ExecutionGraphRunner", "ExecutionMetrics", "ExecutionOp",
    "compile_execution_graph", "compile_residency_regions", "ModelResidencyPlan", "build_model_residency_plan",
    "ModelStreamingExecutor", "StreamingRunMetrics",
    "ResidencyAnalysis", "analyze_plan", "qwen3_5_residency_summary",
    "MaterializationMetrics", "RegionMaterializer", "load_checkpoint_plan",
    "AsyncResidencyManager", "ResidencyMetrics", "QuantizationSpec", "QuantizedTensor",
    "QuantizedMaterializer", "QuantizationRegistry", "quantization_spec_from_config",
    "AdmissionDecision", "ResidencyAdmission",
    "inspect_safetensors_metadata", "RegionSource", "TieredRegionStore", "WeightIndexSource",
]
