import pytest
torch=pytest.importorskip("torch")
from nsa.core.state import CanonicalState
from nsa.cognition.interfaces import ActionCandidate
from nsa.strategy import StrategicController,SystemOneStrategyProvider
from nsa.strategy.torch_adapter import StrategicAttentionBias
def test_torch_attention_bias_broadcasts():
    state=CanonicalState()
    actions=[ActionCandidate("safe",expected_utility=.9,risk=.1),ActionCandidate("fast",expected_utility=.6,risk=.2)]
    field=StrategicController().evaluate(state,SystemOneStrategyProvider().strategic_candidates(state,actions)).field
    logits=torch.zeros(2,4,3,3)
    out=StrategicAttentionBias(.1).apply(logits,field,[{"safe":1.},{"fast":1.},{"safe":.5}])
    assert out.shape==logits.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out,logits)
