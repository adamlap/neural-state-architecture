from nsa.residency.residency_policy import NextUseEvictionPolicy, ResidentEntry


def test_policy_evicts_region_with_no_future_use():
    policy = NextUseEvictionPolicy()
    resident = {
        "state": ResidentEntry("state", 40, next_use=2, priority=10),
        "layer.1": ResidentEntry("layer.1", 40, next_use=None),
    }

    victim = policy.choose_victim(resident, required_bytes=30, capacity_bytes=80)

    assert victim == "layer.1"


def test_policy_protects_high_priority_state():
    policy = NextUseEvictionPolicy()
    resident = {
        "state": ResidentEntry("state", 60, next_use=1, priority=10),
        "layer.1": ResidentEntry("layer.1", 30, next_use=20, priority=0),
    }

    victim = policy.choose_victim(resident, required_bytes=20, capacity_bytes=80)

    assert victim == "layer.1"
