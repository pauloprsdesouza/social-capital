"""Score-based trial re-ranking shared by V2 and V3."""

from __future__ import annotations

import pandas as pd

RECOMMENDATION_COLUMNS = ["user_id", "algorithm", "news_id", "ranking", "rating", "score"]


def empty_recommendations() -> pd.DataFrame:
    return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)


def rerank_by_score(
    trial_items: pd.DataFrame,
    features: pd.DataFrame,
    score_col: str,
    top_k: int = 10,
) -> pd.DataFrame:
    trial = trial_items.copy()
    trial["news_id"] = trial["news_id"].astype(str)
    feat = features.copy()
    feat["news_id"] = feat["news_id"].astype(str)
    if score_col not in feat.columns:
        feat[score_col] = 0.0
    merged = trial.merge(feat[["news_id", score_col]], on="news_id", how="left")
    merged[score_col] = merged[score_col].fillna(0)
    merged = merged.sort_values(score_col, ascending=False).head(top_k).copy()
    merged["ranking"] = range(1, len(merged) + 1)
    return merged


def build_state_art_rows(
    base_ratings: pd.DataFrame,
    features: pd.DataFrame,
    base_algo: str,
    state_art_suffix: str,
    score_col: str = "state_art_score",
    top_k: int = 10,
) -> pd.DataFrame:
    session_rows: list[pd.DataFrame] = []
    for _, grp in base_ratings.groupby(["user_id"], sort=False):
        reranked = rerank_by_score(grp, features, score_col, top_k=top_k)
        reranked["algorithm"] = f"{base_algo}-{state_art_suffix}"
        reranked["score"] = reranked[score_col]
        session_rows.append(reranked[RECOMMENDATION_COLUMNS])
    return pd.concat(session_rows, ignore_index=True) if session_rows else empty_recommendations()


def build_scsa_reranked_rows(
    base_ratings: pd.DataFrame,
    features: pd.DataFrame,
    base_algo: str,
    scsa_suffix: str,
    score_col: str,
    top_k: int = 10,
) -> pd.DataFrame:
    session_rows: list[pd.DataFrame] = []
    for _, grp in base_ratings.groupby(["user_id"], sort=False):
        reranked = rerank_by_score(grp, features, score_col, top_k=top_k)
        reranked["algorithm"] = f"{base_algo}-{scsa_suffix}"
        reranked["score"] = reranked[score_col]
        session_rows.append(reranked[RECOMMENDATION_COLUMNS])
    return pd.concat(session_rows, ignore_index=True) if session_rows else empty_recommendations()


def build_trial_rerank_bundle(
    ratings: pd.DataFrame,
    features: pd.DataFrame,
    *,
    base_algorithms: list[str],
    rerank_suffixes: dict[str, str],
    scsa_score_specs: list[tuple[str, str]],
    state_art_score_col: str = "state_art_score",
    top_k: int = 10,
) -> pd.DataFrame:
    """Build STATE_ART rows plus one or more SCSA re-rank variants per base algorithm."""
    rows: list[pd.DataFrame] = []
    for base in base_algorithms:
        base_ratings = ratings[ratings["algorithm"] == base].copy()
        if base_ratings.empty:
            continue
        rows.append(
            build_state_art_rows(
                base_ratings,
                features,
                base,
                rerank_suffixes["state_art"],
                score_col=state_art_score_col,
                top_k=top_k,
            )
        )
        for suffix_key, score_col in scsa_score_specs:
            part = build_scsa_reranked_rows(
                base_ratings,
                features,
                base,
                rerank_suffixes[suffix_key],
                score_col,
                top_k=top_k,
            )
            if not part.empty:
                rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else empty_recommendations()


def append_suffix_rerank(
    base_recs: pd.DataFrame,
    features: pd.DataFrame,
    score_col: str,
    suffix: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Re-rank each user-algorithm session and append rows with `{algorithm}-{suffix}`."""
    feat = features[["news_id", score_col]].copy()
    feat["news_id"] = feat["news_id"].astype(str)

    reranked_rows: list[pd.DataFrame] = []
    for (_, algo), grp in base_recs.groupby(["user_id", "algorithm"], sort=False):
        merged = grp.merge(feat, on="news_id", how="left")
        merged[score_col] = merged[score_col].fillna(0)
        merged = merged.sort_values(score_col, ascending=False).head(top_k).copy()
        merged["ranking"] = range(1, len(merged) + 1)
        merged["algorithm"] = f"{algo}-{suffix}"
        merged["score"] = merged[score_col]
        reranked_rows.append(merged[RECOMMENDATION_COLUMNS])

    if not reranked_rows:
        return base_recs.copy()
    return pd.concat([base_recs, pd.concat(reranked_rows, ignore_index=True)], ignore_index=True)
