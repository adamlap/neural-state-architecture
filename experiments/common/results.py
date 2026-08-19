"""Common, machine-readable experiment result schema and aggregation helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
import json
import math
import statistics


@dataclass
class ExperimentResult:
    experiment_id: str
    git_commit: str = "unknown"
    seed: int = 0
    model_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    ablation: Dict[str, Any] = field(default_factory=dict)
    perturbation: Dict[str, Any] = field(default_factory=dict)
    invariants: Dict[str, bool] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate scalar metrics across seeds with mean/std and 95% normal CI."""
    metric_values: Dict[str, List[float]] = {}
    for result in results:
        for key, value in result.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metric_values.setdefault(key, []).append(float(value))
    summary: Dict[str, Any] = {"n_seeds": len(results), "metrics": {}}
    for key, values in metric_values.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if values else 0.0
        summary["metrics"][key] = {
            "n": len(values), "mean": mean, "std": std,
            "ci95_low": mean - ci95, "ci95_high": mean + ci95,
        }
    summary["invariants_all_pass"] = all(
        all(bool(v) for v in r.get("invariants", {}).values()) for r in results
    )
    return summary
