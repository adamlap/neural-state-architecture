import pytest

np = pytest.importorskip("numpy")

from nsa import NSA, RuntimeConfig
from nsa.algebra import IntegrityLabel
from nsa.organic import StreamLearner, TrailingField


class _Echo:
    model = "echo"

    def generate(self, prompt, **kw):
        return ""


def test_trailing_field_decays_and_keeps_direction():
    f = TrailingField(8, 0.9)
    cue = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(5):
        f.step(cue)
    before = f.key()
    for _ in range(4):
        f.step(np.zeros(8, dtype=np.float32))
    assert float(before @ f.key()) > 0.99 and float(np.linalg.norm(f.z)) < float(np.linalg.norm(before)) + 1
    with pytest.raises(ValueError):
        TrailingField(8, 1.0)


def _teach(agent_or_learner, trials=60, delay=3, outcome=1.0, rng_seed=0, source=None, blocked=False):
    rng = np.random.default_rng(rng_seed)
    proto = rng.standard_normal(24).astype(np.float32)
    for _ in range(trials):
        for _ in range(4):
            x = proto + 0.3 * rng.standard_normal(24).astype(np.float32)
            (agent_or_learner.observe_signal(x, source=source) if source else agent_or_learner.observe_signal(x))
        for _ in range(delay - 1):
            agent_or_learner.observe_signal(0.2 * rng.standard_normal(24).astype(np.float32))
        kw = {"source": source} if source else {}
        agent_or_learner.observe_signal(0.2 * rng.standard_normal(24).astype(np.float32), outcome=outcome, **kw)
    return proto


def _probe(agent, proto, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    for _ in range(8):
        agent.observe_signal(0.2 * rng.standard_normal(24).astype(np.float32))
    p = 0.0
    for _ in range(4):
        p = agent.observe_signal(proto + 0.3 * rng.standard_normal(24).astype(np.float32))
    return p


def test_runtime_learns_a_delayed_outcome_from_a_continuous_signal_with_a_fixed_size_memory():
    a = NSA(_Echo(), stream_learner=StreamLearner(24, lam=0.8, seed=1))
    proto = _teach(a, trials=80, delay=3, outcome=1.0)
    assert _probe(a, proto) > 0.5 and a.stream_learner.M.shape == (1024,) and a.stream_learner.floats == 1024
    assert a.signal_stats["outcomes"] == 80


def test_soft_state_uncertainty_falls_as_the_learner_stops_being_surprised():
    a = NSA(_Echo(), stream_learner=StreamLearner(24, lam=0.8, seed=2))
    start = a.state.soft.uncertainty
    _teach(a, trials=80)
    assert a.state.soft.uncertainty < 0.2 <= max(start, 0.2)
    assert a.state.soft.confidence == pytest.approx(1.0 - a.stream_learner.uncertainty)


def test_an_untrusted_channel_cannot_teach_the_learner():
    cfg = RuntimeConfig(untrusted_sources=("spoofed",))
    a = NSA(_Echo(), config=cfg, stream_learner=StreamLearner(24, lam=0.8, seed=3))
    proto = _teach(a, trials=80, outcome=1.0, source="spoofed")
    assert _probe(a, proto) == pytest.approx(0.0, abs=1e-6) and a.signal_stats["learning_blocked"] == 80 and a.stream_learner.updates == 0
    t = NSA(_Echo(), config=cfg, stream_learner=StreamLearner(24, lam=0.8, seed=3))
    _teach(t, trials=80, outcome=1.0)                       # the trusted default channel learns
    assert t.stream_learner.updates > 0


def test_explicit_integrity_and_signed_ingress_rule_apply_to_signals():
    a = NSA(_Echo(), stream_learner=StreamLearner(8, seed=1))
    a.observe_signal(np.ones(8, dtype=np.float32), outcome=1.0, integrity=IntegrityLabel.UNTRUSTED)
    assert a.stream_learner.updates == 0
    b = NSA(_Echo(), config=RuntimeConfig(require_signed_ingress=True), stream_learner=StreamLearner(8, seed=1))
    b.observe_signal(np.ones(8, dtype=np.float32), outcome=1.0)
    assert b.stream_learner.updates == 0 and b.signal_stats["learning_blocked"] == 1


def test_signal_api_requires_a_configured_learner():
    with pytest.raises(RuntimeError):
        NSA(_Echo()).observe_signal(np.ones(4, dtype=np.float32))


def test_familiarity_separates_never_seen_inputs_from_inputs_learned_to_mean_nothing():
    rng = np.random.default_rng(0)
    learner = StreamLearner(24, lam=0.8, seed=1)
    seen_proto = rng.standard_normal(24).astype(np.float32)
    new_proto = rng.standard_normal(24).astype(np.float32)

    def present(proto, outcome=None):
        for _ in range(4):
            learner.observe(proto + 0.3 * rng.standard_normal(24).astype(np.float32))
        for _ in range(3):
            learner.observe(0.2 * rng.standard_normal(24).astype(np.float32))
        if outcome is not None:
            learner.learn(outcome)

    for _ in range(20):
        present(seen_proto, outcome=0.0)                       # always "nothing happens": no write, but the input becomes familiar
    present(seen_proto)
    fam_seen, pred_seen = learner.familiarity(), learner.predict()
    present(new_proto)
    assert fam_seen > 0.6 and abs(pred_seen) < 0.2 and learner.familiarity() < fam_seen - 0.3 and learner.updates == 0
