import subprocess
import sys

import pytest

from nsa.core.capabilities import CapabilityAuthority
from tests.test_safety_kernel import create_sample_omega
from nsa.core.safety_kernel import ImmutableSafetyKernel, KernelVerdict
from nsa.core.state import CanonicalState, SemanticState
from nsa.core.state_codec import decode_state, dumps_state, encode_state, round_trip
from nsa.core.transition import state_digest

PROBE = (
    "from nsa.core.state import CanonicalState, SemanticState\n"
    "from nsa.core.transition import state_digest\n"
    "from nsa.core.state_codec import dumps_state\n"
    "s = CanonicalState(semantic=SemanticState({'tags': {'alpha','beta','gamma','delta','epsilon'}, 'n': frozenset({3,1,2})}))\n"
    "print(state_digest(s)); print(dumps_state(s))\n"
)


def _run(seed):
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True,
                         env={"PYTHONHASHSEED": str(seed), "PYTHONPATH": ".", "PATH": ""})
    return out.stdout


def test_digest_and_encoding_do_not_depend_on_the_process_hash_seed():
    """Regression: set-valued semantic state hashed differently per process, breaking journal verification."""
    outputs = {_run(seed) for seed in (1, 2, 3, 4)}
    assert len(outputs) == 1


def test_set_valued_state_round_trips_with_a_verified_digest():
    state = CanonicalState(semantic=SemanticState({"tags": {"b", "a", "c"}}))
    restored = decode_state(encode_state(state))  # decode verifies the digest
    assert sorted(restored.semantic.value["tags"]) == ["a", "b", "c"]
    assert dumps_state(state) == dumps_state(state)
    assert state_digest(round_trip(state)) == state_digest(round_trip(state))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_self_error_is_rejected_by_the_safety_kernel(bad):
    kernel = ImmutableSafetyKernel(CapabilityAuthority(b"k"))
    result = kernel.evaluate_transition(create_sample_omega(), "act", predicted_self_error=bad)
    assert result.verdict == KernelVerdict.ROLLBACK
