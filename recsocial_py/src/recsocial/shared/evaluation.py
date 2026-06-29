"""Evaluation metrics for Social Capital recommender reproduction (SDD §17)."""

from __future__ import annotations

import numpy as np


def is_relevant(rating: float, threshold: int = 4) -> bool:
    return float(rating) >= threshold


def precision_at_k(ratings: list[float], k: int, threshold: int = 4) -> float:
    top = ratings[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if is_relevant(r, threshold))
    return hits / k


def mrr(ratings: list[float], threshold: int = 4) -> float:
    for idx, rating in enumerate(ratings):
        if is_relevant(rating, threshold):
            return 1.0 / (idx + 1)
    return 0.0


def average_precision(ratings: list[float], k: int = 10, threshold: int = 4) -> float:
    """Mean precision at each relevant position (SDD §17.3)."""
    relevant_count = 0
    precision_sum = 0.0
    for idx, rating in enumerate(ratings[:k]):
        if is_relevant(rating, threshold):
            relevant_count += 1
            precision_sum += relevant_count / (idx + 1)
    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count


def dcg_at_k(ratings: list[float], k: int, graded: bool = True) -> float:
    top = ratings[:k]
    if not top:
        return 0.0
    if graded:
        gains = [2 ** float(r) - 1 for r in top]
    else:
        gains = [1.0 if is_relevant(r) else 0.0 for r in top]
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ratings: list[float], k: int = 10, graded: bool = True) -> float:
    dcg = dcg_at_k(ratings, k, graded=graded)
    ideal = sorted((float(r) for r in ratings), reverse=True)[:k]
    idcg = dcg_at_k(ideal, k, graded=graded)
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_metrics_by_algorithm(
    df,
    *,
    user_col: str = "user_id",
    algo_col: str = "algorithm",
    rank_col: str = "ranking",
    rating_col: str = "rating",
    threshold: int = 4,
    k: int = 10,
    graded_ndcg: bool = True,
    algorithm_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute mean metrics per algorithm from a ratings dataframe."""
    import pandas as pd

    data = df.copy()
    if algorithm_aliases:
        data[algo_col] = data[algo_col].map(lambda a: algorithm_aliases.get(a, a))

    data = data.sort_values([user_col, algo_col, rank_col])
    grouped = (
        data.groupby([user_col, algo_col])[rating_col]
        .apply(list)
        .reset_index()
    )
    grouped["mrr"] = grouped[rating_col].apply(lambda r: mrr(r, threshold))
    grouped["map"] = grouped[rating_col].apply(
        lambda r: average_precision(r, k=k, threshold=threshold)
    )
    grouped["ndcg"] = grouped[rating_col].apply(
        lambda r: ndcg_at_k(r, k=k, graded=graded_ndcg)
    )

    summary: dict[str, dict[str, float]] = {}
    for algo, part in grouped.groupby(algo_col):
        summary[str(algo)] = {
            "mrr": float(part["mrr"].mean()),
            "map": float(part["map"].mean()),
            "ndcg": float(part["ndcg"].mean()),
            "n_sessions": int(len(part)),
        }
    return summary
