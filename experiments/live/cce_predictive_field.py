"""Small deterministic experiment for the gated predictive CCE field."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from nsa.runtime.continuous_state_field import ContinuousStateField
from nsa.runtime.predictive_dynamics import StatePredictor
from nsa.runtime.predictive_field import PredictiveDynamicsField


def run(output: str) -> dict:
    torch.manual_seed(7)
    predictor = StatePredictor(2)
    # A known transition gives an auditable integration target without claiming
    # that the toy dynamics are a cognitive model.
    with torch.no_grad():
        for parameter in predictor.parameters():
            parameter.zero_()
        predictor.net[-1].bias.copy_(torch.tensor([1.0, -1.0]))

    field = PredictiveDynamicsField(predictor, reference_dt=0.1, enabled=True)
    state = torch.tensor([[0.0, 0.0]])
    runtime = ContinuousStateField(state, field, enabled=True, integration_cadence_seconds=0.01)
    runtime.step_now(10.0)
    runtime.step_now(10.1)
    after = runtime.state
    runtime.stop()
    result = {
        "predictive_field_enabled": field.enabled,
        "integrations": runtime.status().integration_count,
        "state": after.tolist(),
        "finite": bool(torch.isfinite(after).all()),
        "authority_boundary": "predictive field has no NSA authority access",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/ci/cce/predictive_field.json")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))
