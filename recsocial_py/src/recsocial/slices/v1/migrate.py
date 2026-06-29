"""CSV migration: legacy flat files → SDD-schema CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.slices.v1.config import AppConfig
from recsocial.shared.data_io import build_author_users_from_tweets, load_users_twitter
from recsocial.shared.utils import parse_mentions, parse_topic, safe_str_id


def migrate_to_sdd_schema(cfg: AppConfig) -> dict[str, Path]:
    ratings_raw = pd.read_csv(cfg.paths.raw_ratings)
    tweets = pd.read_csv(cfg.paths.raw_tweets)
    users_twitter = load_users_twitter(Path(cfg.paths.raw_users_twitter))

    rated_ids = set(ratings_raw["item_id"].astype(str))
    tweets = tweets[tweets["Id"].astype(str).isin(rated_ids)].copy()
    news_meta = users_twitter.drop_duplicates("item_id").copy()
    news_meta["item_id"] = news_meta["item_id"].astype(str)
    news_meta = news_meta[news_meta["item_id"].isin(rated_ids)]

    tweets["news_id"] = tweets["Id"].astype(str)
    tweets = tweets.merge(
        news_meta[
            [
                "item_id",
                "text",
                "user_who_published",
                "like_count",
                "retweet_count",
                "reply_count",
                "quote_count",
                "user_followers_count",
                "user_listed_count",
                "user_likes_count",
                "user_tweets_count",
                "user_is_verified",
                "tweet_date",
            ]
        ],
        left_on="news_id",
        right_on="item_id",
        how="left",
    )

    author_stats = build_author_users_from_tweets(tweets)
    if not users_twitter.empty and "user_who_published" in users_twitter.columns:
        twitter_authors = users_twitter.groupby("user_who_published").agg(
            received_likes_count=("like_count", "sum"),
            published_news_count=("item_id", "count"),
            peak_likes=("like_count", "max"),
        ).reset_index()
        twitter_authors["user_id"] = twitter_authors["user_who_published"].astype(str)
        users_df = author_stats.merge(
            twitter_authors[["user_id", "received_likes_count", "published_news_count", "peak_likes"]],
            on="user_id",
            how="outer",
            suffixes=("", "_tw"),
        )
        users_df["received_likes_count"] = users_df[
            ["received_likes_count", "received_likes_count_tw"]
        ].max(axis=1, skipna=True).fillna(0)
        users_df["published_news_count"] = users_df[
            ["published_news_count", "published_news_count_tw"]
        ].max(axis=1, skipna=True).fillna(0)
        users_df["followers_count"] = users_df[["followers_count", "peak_likes"]].max(axis=1).fillna(1)
        users_df = users_df.drop(columns=[c for c in users_df.columns if c.endswith("_tw") or c == "peak_likes"])
    else:
        users_df = author_stats
    for col in (
        "followers_count",
        "lists_count",
        "received_likes_count",
        "published_news_count",
    ):
        users_df[col] = users_df[col].astype(int)

    tweets["mentioned_user_ids"] = tweets["Mentions"].apply(
        lambda x: "|".join(parse_mentions(x))
    )
    tweets["topic"] = tweets["Domains"].apply(parse_topic)
    tweets["subtopic"] = tweets["Entities"].apply(parse_topic)
    news_df = pd.DataFrame(
        {
            "news_id": tweets["news_id"],
            "author_id": tweets["AuthorId"].astype(str),
            "text": tweets["text"].fillna(""),
            "likes_count": tweets["LikeCount"].fillna(0).astype(int),
            "retweets_count": tweets["RetweetCount"].fillna(0).astype(int),
            "comments_count": tweets["ReplyCount"].fillna(0).astype(int),
            "mentioned_user_ids": tweets["mentioned_user_ids"],
            "topic": tweets["topic"],
            "subtopic": tweets["subtopic"],
            "created_at": tweets["CreatedAt"],
            "oracle_social_capital_score": tweets["SocialCapitalScore"],
            "oracle_sentiment_score": tweets["SentimentScore"],
        }
    )

    mention_rows: list[dict[str, str]] = []
    for row in news_df.itertuples(index=False):
        for uid in str(row.mentioned_user_ids).split("|"):
            if uid:
                mention_rows.append(
                    {"source_type": "news", "source_id": row.news_id, "mentioned_user_id": uid}
                )
    mentions_df = pd.DataFrame(mention_rows)
    comments_df = pd.DataFrame(
        columns=[
            "comment_id",
            "news_id",
            "author_id",
            "text",
            "likes_count",
            "retweets_count",
        ]
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": ratings_raw["user_id"].map(safe_str_id),
            "news_id": ratings_raw["item_id"].map(safe_str_id),
            "round_id": 1,
            "position": ratings_raw["ranking"].astype(int),
            "algorithm": ratings_raw["algorithm"].astype(str),
            "rating": ratings_raw["rating"].astype(int),
            "session_id": ratings_raw["id"].astype(int),
        }
    )

    out_dir = Path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "users": out_dir / "users.csv",
        "news": out_dir / "news.csv",
        "comments": out_dir / "comments.csv",
        "mentions": out_dir / "mentions.csv",
        "ratings": out_dir / "ratings.csv",
    }
    users_df.to_csv(paths["users"], index=False)
    news_df.to_csv(paths["news"], index=False)
    comments_df.to_csv(paths["comments"], index=False)
    mentions_df.to_csv(paths["mentions"], index=False)
    ratings_df.to_csv(paths["ratings"], index=False)

    manifest = pd.DataFrame(
        [
            {"dataset": k, "path": str(v), "rows": len(locals()[f"{k}_df"])}
            for k, v in paths.items()
        ]
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    paths["manifest"] = manifest_path
    return paths
