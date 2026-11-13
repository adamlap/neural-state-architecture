"""Choose the opt-in speed/accuracy options from the model that is actually attached.

Every choice here is one that was measured (see research/SPEED_MODEL_CALLS.md and research/RESEARCH_BENCHMARK_V12.md / V13.md); anything unmeasured stays off.

=========================  ==========================  ==============================================================
option                     chosen when                 evidence
=========================  ==========================  ==============================================================
backend ``num_thread``     always (all CPUs)           1.24x (1.5b) / 1.49x (7b) per call, accuracy unchanged
``provenance_window``      always (256)                keeps ``observe()`` cost flat with agent age
``reuse_familiar_answers`` always                      51 % fewer calls, 0 stale reuse (grounded cache)
``answer_from_memory``     always                      deterministic: exact one-key lookups return the stored value, no call
``pin_memory_prefix``      model >= 7B parameters      1.6x at unchanged accuracy on the 7b; 1.00 -> 0.79 on the 1.5b
``retrieve_lines`` = 3     model < 7B parameters       helps the 1.5b when memory misses; slightly hurt the 7b
=========================  ==========================  ==============================================================

A model whose size cannot be detected is treated as small (the conservative side: no pinning). Only fields still at their default are changed, so an explicit
setting always wins.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import fields, replace
from typing import Any, Optional

LARGE_MODEL_B = 7.0          # smallest size on which pinning the whole memory kept accuracy
_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", re.I)


def _parse_size(text: str) -> Optional[float]:
    """'qwen2.5:7b' -> 7.0, 'Qwen2.5-0.5B-Instruct' -> 0.5, '494.03M' -> 0.494; the version number ('2.5') is not a size because no b/m follows it."""
    hits = [(float(n), u.lower()) for n, u in _SIZE.findall(text or "")]
    if not hits:
        return None
    n, unit = hits[-1]
    return n if unit == "b" else n / 1000.0


def detect_model_size(backend: Any, *, timeout: float = 2.0) -> Optional[float]:
    """Parameter count in billions, from the model name, else from Ollama's ``/api/show`` (``details.parameter_size``); None if unknown."""
    inner = getattr(backend, "inner", backend)
    name = str(getattr(inner, "model_name", None) or getattr(inner, "model", None) or getattr(backend, "model", "") or "")
    size = _parse_size(name.split("/")[-1])
    if size is not None:
        return size
    base = getattr(inner, "base_url", None)
    if not base or not name or os.environ.get("NSA_BENCH_NO_GUARD"):
        return None
    try:
        req = urllib.request.Request(f"{base}/api/show", data=json.dumps({"model": name}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _parse_size(str(json.loads(r.read().decode()).get("details", {}).get("parameter_size", "")))
    except Exception:
        return None


def auto_config(backend: Any, base: Any = None) -> Any:
    """Return ``base`` (a RuntimeConfig) with the measured-good options for ``backend``'s model filled into every field that is still at its default."""
    from nsa.agent import RuntimeConfig

    base = base or RuntimeConfig()
    defaults = {f.name: f.default for f in fields(RuntimeConfig)}
    size = detect_model_size(backend)
    large = size is not None and size >= LARGE_MODEL_B
    wanted = {"provenance_window": 256, "reuse_familiar_answers": True, "answer_from_memory": True}
    wanted.update({"pin_memory_prefix": True} if large else {"retrieve_lines": 3})
    changes = {k: v for k, v in wanted.items() if getattr(base, k) == defaults[k]}
    inner = getattr(backend, "inner", backend)
    if hasattr(inner, "num_thread") and not getattr(inner, "num_thread", None) and not getattr(backend, "num_thread", None):
        try:
            inner.num_thread = os.cpu_count() or None
        except Exception:
            pass
    return replace(base, **changes)


__all__ = ["LARGE_MODEL_B", "auto_config", "detect_model_size"]
