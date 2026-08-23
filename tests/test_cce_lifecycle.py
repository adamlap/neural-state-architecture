import json

from nsa.cce.lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore


def test_checkpoint_roundtrip_and_integrity(tmp_path):
    path = tmp_path / "state.json"
    store = StateCheckpointStore(path)
    original = {"working": 0.2, "goal": 0.8}
    envelope = store.save(original)
    restored = store.load()
    assert restored.state == original
    assert restored.state_hash == envelope.state_hash == CheckpointEnvelope.hash_state(original)


def test_checkpoint_detects_tampering(tmp_path):
    path = tmp_path / "state.json"
    StateCheckpointStore(path).save({"goal": 0.5})
    raw = json.loads(path.read_text())
    raw["state"]["goal"] = 0.9
    path.write_text(json.dumps(raw))
    try:
        StateCheckpointStore(path).load()
    except ValueError as exc:
        assert "integrity" in str(exc)
    else:
        raise AssertionError("tampered checkpoint was accepted")


def test_input_queue_preserves_provenance_and_confidence():
    queue = CognitiveInputQueue()
    event = CognitiveInputEvent("hello", source="speech", confidence=0.91, provenance="microphone")
    queue.push(event)
    assert queue.drain() == [event]
    assert len(queue) == 0
