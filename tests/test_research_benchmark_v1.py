from experiments.research_v1.longitudinal_benchmark import extract, make_episode

def test_extract():
    assert extract('VALUE=ABC_12') == 'ABC_12'
    assert extract('answer: VALUE=abc') == 'ABC'
    assert extract('no value') is None

def test_episode_supersession():
    observations, query, expected = make_episode(7, 'supersession', 4)
    assert expected.endswith('CURRENT')
    assert 'obsolete' in observations[1]
    assert 'current value' in query.lower()