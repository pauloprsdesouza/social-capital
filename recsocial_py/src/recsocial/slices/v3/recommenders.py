"""V3 recommenders — SCSA-PLUS re-ranking + PCA suffix (MainV3.ipynb)."""

from __future__ import annotations

import pandas as pd

from recsocial.shared.reranking import (
    append_suffix_rerank,
    build_trial_rerank_bundle,
    empty_recommendations,
    rerank_by_score,
)
from recsocial.slices.v3.config import V3Config


def build_base_recommendations(
    ratings: pd.DataFrame,
    features: pd.DataFrame,
    cfg: V3Config,
) -> pd.DataFrame:
    return build_trial_rerank_bundle(
        ratings,
        features,
        base_algorithms=cfg.base_algorithms,
        rerank_suffixes=cfg.rerank_suffixes,
        scsa_score_specs=[("scsa_plus", "scsa_plus_score")],
    )


def build_standalone_scsa_plus(
    ratings: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, user_ratings in ratings.groupby("user_id", sort=False):
        trial_ids = user_ratings.drop_duplicates("news_id")[["news_id", "user_id"]]
        trial_ids["rating"] = user_ratings.groupby("news_id")["rating"].first().reindex(
            trial_ids["news_id"].values
        ).values
        rec = rerank_by_score(trial_ids, features, "scsa_plus_score", top_k=10)
        rec["algorithm"] = "SCSA_PLUS"
        rec["score"] = rec["scsa_plus_score"]
        rows.append(rec[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]])
    return pd.concat(rows, ignore_index=True) if rows else empty_recommendations()


def build_base_algorithm_trials(ratings: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for base, grp in ratings.groupby("algorithm", sort=False):
        trial = grp.sort_values("position").copy()
        trial["ranking"] = trial["position"]
        trial["score"] = 0.0
        rows.append(trial[["user_id", "algorithm", "news_id", "ranking", "rating", "score"]])
    return pd.concat(rows, ignore_index=True) if rows else empty_recommendations()


def build_v3_recommendations(
    ratings: pd.DataFrame,
    features: pd.DataFrame,
    cfg: V3Config,
) -> pd.DataFrame:
    base = build_base_recommendations(ratings, features, cfg)
    combined = append_suffix_rerank(
        base,
        features,
        score_col="pca1_score",
        suffix=cfg.rerank_suffixes["scsa_plus_v3"],
    )
    parts = [build_base_algorithm_trials(ratings), combined, build_standalone_scsa_plus(ratings, features)]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)
