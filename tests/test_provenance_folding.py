import time

from nsa import NSA, RuntimeConfig
from nsa.core.state import CanonicalState, ProvenanceState
from nsa.core.state_codec import decode_state, encode_state
from nsa.core.transition import state_digest


class _Null:
    model = "n"

    def generate(self, prompt, **kw):
        return ""


def test_never_folded_states_keep_the_original_digest_payload():
    from nsa.core.transition import _canonical_payload

    state = CanonicalState().__class__()
    assert "folded" not in _canonical_payload(state)["provenance"]
    with_history = ProvenanceState(sources=("a", "b"))
    assert ProvenanceState().fold(None) == ProvenanceState() and with_history.fold(100) is with_history      # nothing to fold


def test_fold_keeps_the_window_is_amortised_and_commits_to_what_it_dropped():
    p = ProvenanceState()
    for i in range(40):
        p = p.extend(source=f"s{i}").fold(10)
    assert len(p.sources) <= 20 and p.sources[-1] == "s39" and p.folded_count + len(p.sources) == 40 and p.folded_chain
    # tamper-evident: the same retained tail with a different dropped history gives a different chain
    q = ProvenanceState()
    for i in range(40):
        q = q.extend(source=("X" if i == 3 else f"s{i}")).fold(10)
    assert q.sources == p.sources and q.folded_chain != p.folded_chain
    s1 = CanonicalState().__class__(provenance=p)
    s2 = CanonicalState().__class__(provenance=q)
    assert state_digest(s1) != state_digest(s2)
    # deterministic
    r = ProvenanceState()
    for i in range(40):
        r = r.extend(source=f"s{i}").fold(10)
    assert r.folded_chain == p.folded_chain and state_digest(CanonicalState().__class__(provenance=r)) == state_digest(s1)


def test_folded_state_round_trips_through_the_codec():
    p = ProvenanceState()
    for i in range(50):
        p = p.extend(source=f"s{i}").fold(8)
    state = CanonicalState().__class__(provenance=p)
    back = decode_state(encode_state(state))
    assert back.provenance.folded_count == p.folded_count and back.provenance.folded_chain == p.folded_chain and state_digest(back) == state_digest(state)


def test_runtime_with_a_window_keeps_state_bounded_and_per_call_cost_flat():
    def per_call(window, n_warm, n_measure=300):
        a = NSA(_Null(), config=RuntimeConfig(provenance_window=window))
        for i in range(n_warm):
            a.observe(f"w{i}", source="chat")
        best = float("inf")
        for rep in range(3):                                            # best of three bursts: a busy machine must not fail the test
            t = time.perf_counter()
            for i in range(n_measure // 3):
                a.observe(f"m{rep}-{i}", source="chat")
            best = min(best, (time.perf_counter() - t) / (n_measure // 3))
        return best, a

    young, _ = per_call(64, 200)
    old, agent = per_call(64, 6000)
    assert len(agent.state.provenance.sources) <= 2 * 64 + 5 and agent.state.provenance.folded_count > 5000
    assert old < 3 * young + 5e-4                                      # flat with age (unfolded, 6000 observations cost ~15x more per call)
    assert agent._bounded_state_summary()["source_count"] >= 6300
