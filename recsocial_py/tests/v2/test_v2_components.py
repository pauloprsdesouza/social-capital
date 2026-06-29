"""V2 component and scorer tests."""

from __future__ import annotations

import pandas as pd
import pytest

from recsocial.slices.v1.config import TextConfig
from recsocial.slices.v2.components import (
    compute_components,
    sentiment_impact_from_compound,
    weighted_social_capital,
)
from recsocial.slices.v2.config import RecencyConfig, SocialCapitalV2Config, SocialCapitalWeights


def test_sentiment_impact_range():
    assert sentiment_impact_from_compound(-1) == 0.0
    assert sentiment_impact_from_compound(1) == 1.0


def test_weights_sum_to_one():
    w = SocialCapitalWeights().model_dump()
    core = sum(w[k] for k in (
        "sentiment_impact", "engagement_score", "content_relevance",
        "network_influence", "author_influence", "content_virality",
    ))
    assert abs(core - 1.0) < 1e-9


def test_social_capital_increases_with_engagement():
    row_low = pd.Series({
        "sentiment_impact": 0.5, "engagement_score": 0.1, "content_relevance": 0.1,
        "network_influence": 0.1, "author_influence": 0.1, "content_virality": 0.1,
        "recency_score": 0.0, "diversity_score": 0.0, "context_score": 0.0,
    })
    row_high = row_low.copy()
    row_high["engagement_score"] = 0.9
    weights = SocialCapitalWeights().model_dump()
    assert weighted_social_capital(row_high, weights, False) > weighted_social_capital(row_low, weights, False)


def test_compute_components_minimal():
    news = pd.DataFrame({
        "news_id": ["n1", "n2"],
        "author_id": ["u1", "u2"],
        "text": ["great launch today", "sad news about sports"],
        "likes_count": [100, 5],
        "retweets_count": [50, 1],
        "comments_count": [10, 0],
        "quote_count": [5, 0],
        "sentiment_score": [0.5, -0.5],
        "diversity_score": [0.5, 0.5],
        "context_score": [0.1, 0.1],
        "created_at": ["2021-03-01", "2021-03-02"],
        "mentions_raw": ["", "{123}"],
        "urls_raw": ["{'http://x.com'}", ""],
    })
    users = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "followers_count": [1000, 100],
        "lists_count": [10, 5],
        "received_likes_count": [500, 50],
        "published_news_count": [100, 10],
        "is_verified": [1, 0],
    })
    frame = compute_components(
        news, users, TextConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
        SocialCapitalV2Config(), RecencyConfig(),
    )
    assert "social_capital_v2" not in frame.df.columns
    assert "engagement_score" in frame.df.columns
    assert frame.df["mentions_count"].iloc[1] == 1
