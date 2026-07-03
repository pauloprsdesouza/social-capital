"""V2 recommenders and re-ranking (SDD §12)."""

from __future__ import annotations

import pandas as pd

from recsocial.shared.reranking import build_trial_rerank_bundle, empty_recommendations
from recsocial.slices.v2.config import V2Config


def _score_column(algorithm: str) -> str:
    return {
        "SCSA_PLUS_V3": "social_capital_v2",
        "SCSA_PLUS": "scsa_v1",
        "STATE_ART": "state_art_score",
        "B1": "b1_score",
        "CS": "content_relevance",
        "SC": "social_capital_v2",
    }.get(algorithm, "social_capital_v2")


def build_reranked_recommendations(
    ratings: pd.DataFrame,
    components: pd.DataFrame,
    cfg: V2Config,
    v1_scsa: pd.DataFrame | None = None,
) -> pd.DataFrame:
    comp_df = components.copy()
    if v1_scsa is not None:
        comp_df = comp_df.merge(
            v1_scsa[["news_id", "social_capital_score"]].rename(
                columns={"social_capital_score": "scsa_v1"}
            ),
            on="news_id",
            how="left",
        )
        comp_df["scsa_v1"] = comp_df["scsa_v1"].fillna(0)

    scsa_specs = [
        ("scsa_plus", _score_column("SCSA_PLUS")),
        ("scsa_plus_v3", _score_column("SCSA_PLUS_V3")),
    ]
    result = build_trial_rerank_bundle(
        ratings,
        comp_df,
        base_algorithms=cfg.base_algorithms,
        rerank_suffixes=cfg.rerank_suffixes,
        scsa_score_specs=scsa_specs,
        top_k=cfg.evaluation.ndcg_k,
    )
    return result if not result.empty else empty_recommendations()
