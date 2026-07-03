"""Paired t-tests and exploratory analysis (SDD §23–24)."""

from __future__ import annotations

import pandas as pd

__all__ = ["correlation_matrix", "ranking_shift_analysis"]


def correlation_matrix(features: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in [
            "diversity_score",
            "sentiment_score",
            "scsa_plus_score",
            "social_capital_v2",
            "comments_count",
            "recency_score_v3",
            "quote_count",
            "retweets_count",
            "engagement_score",
            "context_score_v3",
            "likes_count",
            "pca1_score",
        ]
        if c in features.columns
    ]
    if not cols:
        return pd.DataFrame()
    return features[cols].corr()


def ranking_shift_analysis(
    original: pd.DataFrame,
    reranked: pd.DataFrame,
) -> pd.DataFrame:
    """ranking_difference = old_rank - new_rank (positive = moved up)."""
    o = original.rename(columns={"ranking": "old_rank", "position": "old_rank"})
    r = reranked.rename(columns={"ranking": "new_rank"})
    if "old_rank" not in o.columns and "position" in o.columns:
        o["old_rank"] = o["position"]
    merged = o.merge(
        r[["user_id", "algorithm", "news_id", "new_rank", "score"]],
        on=["user_id", "news_id"],
        how="inner",
        suffixes=("_orig", "_new"),
    )
    if "algorithm_orig" in merged.columns:
        merged["algorithm"] = merged["algorithm_orig"]
    elif "algorithm" not in merged.columns:
        merged["algorithm"] = reranked["algorithm"].iloc[0] if len(reranked) else ""
    merged["ranking_difference"] = merged["old_rank"] - merged["new_rank"]
    return merged
