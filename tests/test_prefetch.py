from nsa.residency.prefetch import SequentialPrefetcher


def test_prefetcher_uses_sequential_prior_on_cold_start():
    predictor = SequentialPrefetcher()

    decisions = predictor.predict("layer.0", ["layer.1", "layer.2"], top_k=1)

    assert decisions[0].region == "layer.1"
    assert decisions[0].confidence == 0.5


def test_prefetcher_learns_observed_transition():
    predictor = SequentialPrefetcher()
    predictor.observe("layer.0", "layer.3")
    predictor.observe("layer.0", "layer.3")
    predictor.observe("layer.0", "layer.1")

    decisions = predictor.predict("layer.0", ["layer.1", "layer.3"], top_k=2)

    assert decisions[0].region == "layer.3"
    assert decisions[0].confidence == 2 / 3
    assert decisions[1].region == "layer.1"
