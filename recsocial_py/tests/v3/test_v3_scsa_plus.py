"""V3 SCSA-PLUS and PCA tests."""

from __future__ import annotations

import pandas as pd

from recsocial.shared.reranking import append_suffix_rerank, rerank_by_score
from recsocial.slices.v3.config import ContextConfig, PcaConfigV3, RecencyV3Config, SocialCapitalV3Config
from recsocial.slices.v3.pca_ranking import combined_text, compute_pca_scores
from recsocial.slices.v3.tweet_metrics import recency_score_v3, scsa_plus_score


def test_recency_decreases_with_age():
    from datetime import datetime, timezone

    ref = datetime(2021, 6, 1, tzinfo=timezone.utc)
    fresh = recency_score_v3("2021-05-31", ref, decay_factor=0.1)
    stale = recency_score_v3("2020-01-01", ref, decay_factor=0.1)
    assert fresh > stale


def test_scsa_plus_includes_engagement():
    cfg = SocialCapitalV3Config()
    low = pd.Series({"retweets_count": 0, "likes_count": 0, "comments_count": 0})
    high = pd.Series({"retweets_count": 100, "likes_count": 50, "comments_count": 10})
    assert scsa_plus_score(high, 1.0, 0.0, 1.0, cfg) > scsa_plus_score(low, 1.0, 0.0, 1.0, cfg)


def test_combined_text_merges_tokens():
    row = pd.Series({"text": "hello", "tokens_raw": "{'world'}", "hashtags_raw": "{'news'}"})
    assert "hello" in combined_text(row)
    assert "world" in combined_text(row)


def test_pca_scores_column():
    df = pd.DataFrame({
        "news_id": ["n1", "n2"],
        "text": ["alpha beta", "gamma delta"],
        "tokens_raw": ["", ""],
        "hashtags_raw": ["", ""],
        "sentiment_score": [0.5, -0.2],
        "diversity_score": [0.3, 0.4],
        "context_score": [0.1, 0.2],
        "likes_count": [10, 5],
        "comments_count": [2, 1],
        "quote_count": [0, 0],
        "retweets_count": [3, 1],
        "scsa_plus_score": [1.0, 0.5],
        "social_capital_v2": [0.8, 0.6],
    })
    out = compute_pca_scores(df, PcaConfigV3(enabled=True, variance_ratio=0.95, max_features=50))
    assert "pca1_score" in out.columns
    assert out["pca1_score"].notna().all()


def test_append_pca_suffix():
    base = pd.DataFrame({
        "user_id": [1, 1],
        "algorithm": ["B1-STATE_ART", "B1-STATE_ART"],
        "news_id": ["a", "b"],
        "ranking": [1, 2],
        "rating": [5, 3],
        "score": [0.1, 0.2],
    })
    features = pd.DataFrame({
        "news_id": ["a", "b"],
        "pca1_score": [0.9, 0.1],
    })
    out = append_suffix_rerank(base, features, "pca1_score", "SCSA_PLUS_V3")
    algos = set(out["algorithm"])
    assert "B1-STATE_ART-SCSA_PLUS_V3" in algos
    assert "B1-STATE_ART" in algos


def test_rerank_by_score_orders_descending():
    trial = pd.DataFrame({"news_id": ["a", "b", "c"], "user_id": [1, 1, 1], "rating": [3, 5, 4]})
    features = pd.DataFrame({"news_id": ["a", "b", "c"], "scsa_plus_score": [0.1, 0.9, 0.5]})
    out = rerank_by_score(trial, features, "scsa_plus_score", top_k=3)
    assert out.iloc[0]["news_id"] == "b"
