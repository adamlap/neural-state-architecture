"""Checkpoint/restart evidence for CCE lifecycle state."""
from __future__ import annotations

import argparse
import json

from nsa.cce.lifecycle import StateCheckpointStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cce_checkpoint_recovery.json")
    parser.add_argument("--checkpoint", default="results/cce_state.json")
    args = parser.parse_args()

    original = {"working": 0.4, "self_state": 0.2, "goal": 0.9, "uncertainty": 0.1}
    store = StateCheckpointStore(args.checkpoint)
    saved = store.save(original)
    restored = store.load()
    evidence = {
        "schema_version": saved.schema_version,
        "round_trip_equal": restored.state == original,
        "integrity_verified": restored.state_hash == saved.state_hash,
        "checkpoint_exists": store.exists(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)
    print(json.dumps(evidence, indent=2))
    return 0 if all(evidence.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
