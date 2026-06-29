"""Load canonical raw CSV inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_users_twitter(path: str | Path) -> pd.DataFrame:
    """Load tweet/user metadata; supports semicolon legacy or comma export formats."""
    path = Path(path)
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(path, delimiter=delimiter)
    df["item_id"] = df["item_id"].astype(str)

    defaults = {
        "user_followers_count": 0,
        "user_listed_count": 0,
        "user_likes_count": 0,
        "user_tweets_count": 0,
        "user_is_verified": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    if "tweet_date" not in df.columns and "date" in df.columns:
        df["tweet_date"] = df["date"]
    if "tweet_date" not in df.columns:
        df["tweet_date"] = ""

    for col in ("like_count", "retweet_count", "reply_count", "quote_count"):
        if col not in df.columns:
            df[col] = 0

    return df


def build_author_users_from_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    """Derive author user metrics from tweet engagement when Twitter profile CSV lacks counts."""
    frame = tweets.copy()
    frame["author_id"] = frame["AuthorId"].astype(str)
    for col, default in (
        ("LikeCount", 0),
        ("RetweetCount", 0),
        ("ReplyCount", 0),
        ("QuoteCount", 0),
        ("EngagementScore", 0),
    ):
        if col not in frame.columns:
            frame[col] = default
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    grouped = frame.groupby("author_id", as_index=False).agg(
        received_likes_count=("LikeCount", "sum"),
        published_news_count=("author_id", "count"),
        peak_engagement=("EngagementScore", "max"),
        peak_likes=("LikeCount", "max"),
    )
    grouped["followers_count"] = grouped["peak_likes"].clip(lower=1).astype(int)
    grouped["lists_count"] = grouped["published_news_count"].clip(lower=1).astype(int)
    grouped["is_verified"] = 0
    grouped["user_id"] = grouped["author_id"]
    return grouped[
        [
            "user_id",
            "followers_count",
            "lists_count",
            "received_likes_count",
            "published_news_count",
            "is_verified",
        ]
    ]
