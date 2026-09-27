from nsa.residency.admission import ResidencyAdmission


def test_admission_accepts_when_capacity_exists():
    decision = ResidencyAdmission(100).decide(60, 30)
    assert decision.admitted is True
    assert decision.evictions_required == 0


def test_admission_rejects_region_larger_than_tier():
    decision = ResidencyAdmission(100).decide(0, 101)
    assert decision.admitted is False
    assert decision.evictions_required == 0


def test_admission_requests_eviction_under_pressure():
    decision = ResidencyAdmission(100).decide(90, 30)
    assert decision.admitted is True
    assert decision.evictions_required == 1
