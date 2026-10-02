from time import monotonic

from nsa.residency.controller import ActiveResidencyController
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyPolicy
from nsa.residency.types import ResidencyEvent


def _event(region, action, timestamp, latency=0.0, reason="test"):
    return ResidencyEvent(
        timestamp=timestamp,
        region_id=region,
        action=action,
        source=None,
        destination=None,
        bytes_moved=0,
        latency_ms=latency,
        reason=reason,
    )


def test_adaptive_lookahead_grows_when_prefetch_has_useful_lead():
    manager = NeuralResidencyManager(ResidencyPolicy(1024, 4096))
    controller = ActiveResidencyController(
        manager,
        load_fn=lambda *_: None,
        lookahead=1,
        adaptive_lookahead=True,
        min_lookahead=1,
        max_lookahead=3,
    )
    now = monotonic()
    manager.record_event(_event("b", "prefetch", now))
    manager.record_event(_event("b", "prefetch-complete", now + 0.01))
    manager.record_event(_event("b", "execute", now + 0.10, latency=1.0))
    controller._adapt_from_recent_trace()
    assert controller.lookahead == 2
    controller.shutdown()


def test_adaptive_lookahead_shrinks_when_prefetch_misses():
    manager = NeuralResidencyManager(ResidencyPolicy(1024, 4096))
    controller = ActiveResidencyController(
        manager,
        load_fn=lambda *_: None,
        lookahead=3,
        adaptive_lookahead=True,
        min_lookahead=1,
        max_lookahead=3,
    )
    now = monotonic()
    manager.record_event(_event("b", "prefetch", now))
    manager.record_event(_event("b", "execute", now + 0.01, latency=1.0))
    controller._adapt_from_recent_trace()
    assert controller.lookahead == 2
    controller.shutdown()
