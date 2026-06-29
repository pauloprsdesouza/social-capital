"""Paper-aligned metrics (legacy results-social-capital.ipynb + FedCSIS MAP)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsocial.shared.evaluation import is_relevant, precision_at_k


def mrr_by_rank_column(
    session: pd.DataFrame,
    *,
    rank_col: str = "ranking",
    rating_col: str = "rating",
    threshold: int = 4,
) -> float:
    """MRR using min rank among relevant items (multi-item rank slots)."""
    relevant = session[session[rating_col].astype(float) >= threshold]
    if relevant.empty:
        return float("nan")
    return 1.0 / float(relevant[rank_col].astype(float).min())


def map_by_rank_column(
    session: pd.DataFrame,
    *,
    rank_col: str = "ranking",
    rating_col: str = "rating",
    k: int = 10,
    threshold: int = 4,
) -> float:
    """MAP@K from legacy notebook (precision cumsum / rank column)."""
    frame = session.sort_values(rank_col).copy()
    frame["is_relevant"] = frame[rating_col].astype(float) >= threshold
    frame["precision_at_rank"] = (
        frame.groupby(level=0)["is_relevant"].cumsum() / frame[rank_col].astype(float)
        if frame.index.nlevels > 1
        else frame["is_relevant"].cumsum() / frame[rank_col].astype(float)
    )
    top = frame[frame[rank_col].astype(float) <= k]
    hits = top[top["is_relevant"]]
    if hits.empty:
        return 0.0
    return float(hits["precision_at_rank"].mean())


def ndcg_binary_by_rank_column(
    session: pd.DataFrame,
    *,
    rank_col: str = "ranking",
    rating_col: str = "rating",
    k: int = 10,
    threshold: int = 4,
) -> float:
    """Binary NDCG@K using rank column (legacy notebook)."""
    frame = session.sort_values(rank_col)
    top = frame[frame[rank_col].astype(float) <= k]
    relevance = (top[rating_col].astype(float) >= threshold).astype(int).to_numpy()
    if relevance.size == 0:
        return 0.0
    dcg = np.sum(relevance / np.log2(np.arange(2, relevance.size + 2)))
    ideal = np.sort(relevance)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, ideal.size + 2)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def map_fedcsis_pooled(ratings: list[float], k: int = 10, threshold: int = 4) -> float:
    """MAP@K for FedCSIS when sessions pool multiple items per rank slot."""
    rel_in_k = 0
    precision_sum = 0.0
    for idx, rating in enumerate(ratings[:k]):
        if is_relevant(rating, threshold):
            rel_in_k += 1
            precision_sum += rel_in_k / (idx + 1)
    if rel_in_k == 0:
        return 0.0
    return precision_sum / (0.75 * rel_in_k + 0.25 * k)


def evaluate_session_paper_notebook(
    session: pd.DataFrame,
    settings,
) -> dict[str, float]:
    rank_col = settings.rank_col
    rating_col = settings.rating_col
    k = settings.ndcg_k
    threshold = settings.relevance_threshold

    row: dict[str, float] = {
        "mrr": mrr_by_rank_column(
            session, rank_col=rank_col, rating_col=rating_col, threshold=threshold
        ),
        "map": map_by_rank_column(
            session, rank_col=rank_col, rating_col=rating_col, k=k, threshold=threshold
        ),
        "ndcg": ndcg_binary_by_rank_column(
            session, rank_col=rank_col, rating_col=rating_col, k=k, threshold=threshold
        ),
    }
    ordered = session.sort_values(rank_col)[rating_col].astype(float).tolist()
    for pk in settings.precision_k_values:
        row[f"precision_at_{pk}"] = precision_at_k(ordered, pk, threshold)
    return row
