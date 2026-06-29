"""V3 user metrics: reputation, influence, user strength (SDD §11–13)."""

from __future__ import annotations

import math

import pandas as pd

from recsocial.slices.v3.config import InfluenceV3Config, ReputationConfig


def reputation_score_fallback(user_row: pd.Series, cfg: ReputationConfig) -> float:
    """When mention/reply texts are unavailable, use listed-count normalization."""
    theta = cfg.listed_count_theta
    listed = float(user_row.get("lists_count", user_row.get("listed_count", 0)))
    base = 0.0
    return (base + theta * listed) / (1.0 + theta * listed)


def influence_score_v3(
    user_row: pd.Series,
    author_tweets: pd.DataFrame,
    cfg: InfluenceV3Config,
) -> float:
    followers = max(int(user_row.get("followers_count", 0)), 0)
    if followers == 0 and cfg.zero_followers_strategy == "safe_one":
        followers = 1
    recent = author_tweets.head(cfg.max_recent_tweets)
    tweet_count = len(recent)
    if tweet_count == 0 or followers == 0:
        engagement = 0.0
    else:
        total = (
            recent["likes_count"].sum()
            + recent["retweets_count"].sum()
            + recent.get("quote_count", pd.Series(0, index=recent.index)).sum()
            + recent["comments_count"].sum()
        )
        if cfg.include_impressions and "impression_count" in recent.columns:
            total += recent["impression_count"].sum()
        engagement = total / (tweet_count * followers)
    return math.log(followers + 1, cfg.log_base) * (engagement + 1.0)


def user_strength_score(reputation: float, influence: float) -> float:
    return reputation * influence


def build_user_strength_table(
    users_df: pd.DataFrame,
    news_df: pd.DataFrame,
    rep_cfg: ReputationConfig,
    inf_cfg: InfluenceV3Config,
) -> pd.DataFrame:
    news = news_df.copy()
    news["author_id"] = news["author_id"].astype(str)
    users = users_df.copy()
    users["user_id"] = users["user_id"].astype(str)

    rows = []
    for _, urow in users.iterrows():
        uid = str(urow["user_id"])
        author_news = news[news["author_id"] == uid]
        rep = reputation_score_fallback(urow, rep_cfg)
        inf = influence_score_v3(urow, author_news, inf_cfg)
        rows.append(
            {
                "user_id": uid,
                "reputation_score": rep,
                "influence_score_v3": inf,
                "user_strength_score": user_strength_score(rep, inf),
            }
        )
    return pd.DataFrame(rows)
