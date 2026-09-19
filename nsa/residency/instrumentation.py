"""Runtime instrumentation for transformer-region residency."""
from __future__ import annotations
from functools import wraps
from time import monotonic
from typing import Any, Callable, Optional
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.types import MemoryTier, ResidencyEvent

def instrument_decoder_layers(model: Any, manager: NeuralResidencyManager, on_region: Optional[Callable[[str], None]] = None) -> int:
    """Wrap decoder layer forwards so actual region execution feeds the predictor.

    Accelerate's disk hooks run before the wrapped forward, so the event is
    emitted at the point where the region is about to execute. This does not
    fabricate a residency transition: it records execution against the
    region-level control plane while Accelerate owns tensor materialization.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0
    count = 0
    for index, layer in enumerate(layers):
        region_id = f"layer.{index}"
        if region_id not in manager.regions or getattr(layer, "_nsa_residency_wrapped", False):
            continue
        original = layer.forward
        @wraps(original)
        def wrapped(*args, __original=original, __rid=region_id, **kwargs):
            started = monotonic()
            previous = manager.current_region
            if hasattr(manager.predictor, "observe_transition"):
                manager.predictor.observe_transition(previous, __rid)
            manager.current_region = __rid
            if on_region is not None:
                on_region(__rid)
            result = __original(*args, **kwargs)
            manager.record_event(ResidencyEvent(
                monotonic(), __rid, "execute", manager.tiers.get(__rid, MemoryTier.NVME),
                manager.tiers.get(__rid, MemoryTier.NVME), 0,
                (monotonic()-started)*1000, "decoder-forward"
            ))
            return result
        layer.forward = wrapped
        layer._nsa_residency_wrapped = True
        count += 1
    return count
