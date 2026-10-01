"""Admission decisions for heterogeneous memory tiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    evictions_required: int
    reason: str


class ResidencyAdmission:
    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes

    def decide(self, resident_bytes: int, requested_bytes: int) -> AdmissionDecision:
        if requested_bytes > self.capacity_bytes:
            return AdmissionDecision(False, 0, "region exceeds tier capacity")
        deficit = resident_bytes + requested_bytes - self.capacity_bytes
        if deficit <= 0:
            return AdmissionDecision(True, 0, "capacity available")
        return AdmissionDecision(
            True,
            1,
            f"free at least {deficit} bytes through policy eviction",
        )
