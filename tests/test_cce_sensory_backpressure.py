"""Tests for BoundedEventQueue and Backpressure Policies (Phase CCE-6)."""
from __future__ import annotations

import pytest
from nsa.runtime.cce_sensory import (
    BackpressurePolicy,
    BoundedEventQueue,
    CCESensoryIngress,
    SensoryBackpressureError,
    SensoryEvent,
)


def test_bounded_queue_drop_oldest_policy():
    q = BoundedEventQueue(max_size=3, policy=BackpressurePolicy.DROP_OLDEST)
    e1 = SensoryEvent(source="s1", content="one", timestamp_utc=1.0)
    e2 = SensoryEvent(source="s2", content="two", timestamp_utc=2.0)
    e3 = SensoryEvent(source="s3", content="three", timestamp_utc=3.0)
    e4 = SensoryEvent(source="s4", content="four", timestamp_utc=4.0)

    q.push(e1)
    q.push(e2)
    q.push(e3)
    assert q.size == 3
    assert q.dropped_count == 0

    # Pushing 4th drops oldest (e1)
    q.push(e4)
    assert q.size == 3
    assert q.dropped_count == 1

    popped = q.pop_batch(max_count=10)
    assert len(popped) == 3
    assert popped[0].content == "two"
    assert popped[2].content == "four"


def test_bounded_queue_reject_new_policy():
    q = BoundedEventQueue(max_size=2, policy=BackpressurePolicy.REJECT_NEW)
    e1 = SensoryEvent(source="s1", content="one", timestamp_utc=1.0)
    e2 = SensoryEvent(source="s2", content="two", timestamp_utc=2.0)
    e3 = SensoryEvent(source="s3", content="three", timestamp_utc=3.0)

    q.push(e1)
    q.push(e2)

    with pytest.raises(SensoryBackpressureError):
        q.push(e3)

    assert q.dropped_count == 1
