"""Provider-neutral learning surfaces over complete cognitive trajectories.

The learner intentionally does not train a model or grant authority. It turns
committed CCE trajectories into deterministic transition examples and outcome
statistics that higher-level training systems can consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from nsa.cce.trajectory import CognitiveTrajectory


@dataclass(frozen=True)
class TrajectoryExample:
    """One causal training/evaluation example extracted from a trajectory."""

    step: int
    previous_digest: str
    next_digest: str
    action_id: str | None
    committed: bool
    outcome: str
    events: tuple[str, ...]


class TrajectoryLearner:
    """Extract outcome-aware examples without changing canonical state."""

    def examples(self, trajectory: CognitiveTrajectory) -> tuple[TrajectoryExample, ...]:
        records = trajectory.records
        examples: list[TrajectoryExample] = []
        for previous, current in zip(records, records[1:]):
            receipt = current.receipt
            action_id = receipt.action_id if receipt is not None else None
            committed = bool(receipt and receipt.committed)
            outcome = "committed" if committed else "rejected"
            events = tuple(event.kind.value for event in current.events)
            examples.append(TrajectoryExample(
                step=current.step,
                previous_digest=previous.state_digest,
                next_digest=current.state_digest,
                action_id=action_id,
                committed=committed,
                outcome=outcome,
                events=events,
            ))
        return tuple(examples)

    def action_outcomes(self, trajectory: CognitiveTrajectory) -> Mapping[str, Mapping[str, float]]:
        """Return empirical counts/rates for actions represented in a trajectory."""
        totals: dict[str, list[int]] = {}
        for example in self.examples(trajectory):
            if not example.action_id:
                continue
            bucket = totals.setdefault(example.action_id, [0, 0])
            bucket[0] += 1
            bucket[1] += int(example.committed)
        return {
            action: {
                "attempts": float(values[0]),
                "committed": float(values[1]),
                "commit_rate": values[1] / values[0] if values[0] else 0.0,
            }
            for action, values in totals.items()
        }

    def training_records(self, trajectories: Iterable[CognitiveTrajectory]) -> tuple[TrajectoryExample, ...]:
        """Flatten complete trajectories while preserving their causal order."""
        result: list[TrajectoryExample] = []
        for trajectory in trajectories:
            result.extend(self.examples(trajectory))
        return tuple(result)

    @staticmethod
    def event_counts(trajectory: CognitiveTrajectory) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for event in trajectory.events():
            key = event.kind.value
            counts[key] = counts.get(key, 0) + 1
        return counts


__all__ = ["TrajectoryExample", "TrajectoryLearner"]
