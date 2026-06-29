from recsocial.slices.v1.influence import influence_score, popularity_score, reputation_score


def test_popularity_score_bounds():
    assert 0.0 <= popularity_score(0) < popularity_score(1000) <= 1.0


def test_reputation_zero_lists():
    assert reputation_score(100, 0) == 100.0


def test_influence_pseudocode_zero_followers():
    score = influence_score(
        0, 10, 5, 0,
        mode="paper_pseudocode",
        max_followers=1000,
    )
    assert score > 0
