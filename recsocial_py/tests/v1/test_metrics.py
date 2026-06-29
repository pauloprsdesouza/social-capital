from recsocial.shared.evaluation import (
    average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
)


def test_mrr_first_relevant_at_rank_2():
    ratings = [2, 4, 5]
    assert mrr(ratings) == 0.5


def test_map_two_relevant_items():
  ratings = [4, 2, 5, 1, 4]
  ap = average_precision(ratings, k=5)
  # relevant at 1 and 3 and 5 -> AP = (1/1 + 2/3 + 3/5) / 3
  expected = ((1 / 1) + (2 / 3) + (3 / 5)) / 3
  assert abs(ap - expected) < 1e-9


def test_ndcg_perfect_order():
    ratings = [5, 4, 3, 2, 1]
    assert abs(ndcg_at_k(ratings, k=5) - 1.0) < 1e-9


def test_precision_at_5():
    ratings = [1, 2, 4, 5, 3, 4]
    assert precision_at_k(ratings, 5) == 0.4
