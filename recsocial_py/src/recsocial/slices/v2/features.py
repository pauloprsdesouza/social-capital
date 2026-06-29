"""V2 CSV enrichment and component scoring pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.slices.v1.config import AppConfig
from recsocial.slices.v1.migrate import migrate_to_sdd_schema
from recsocial.shared.data_io import load_users_twitter
from recsocial.shared.utils import parse_mentions
from recsocial.slices.v2.components import ComponentFrame, compute_components, weighted_social_capital
from recsocial.slices.v2.config import V2Config


def enrich_news_for_v2(cfg: V2Config, v1_cfg: AppConfig) -> pd.DataFrame:
    processed_v1 = Path(v1_cfg.paths.processed_dir)
    if not (processed_v1 / "news.csv").exists():
        migrate_to_sdd_schema(v1_cfg)

    news = pd.read_csv(processed_v1 / "news.csv")
    news["news_id"] = news["news_id"].astype(str)
    tweets = pd.read_csv(v1_cfg.paths.raw_tweets)
    users_twitter = load_users_twitter(v1_cfg.paths.raw_users_twitter)

    rated_ids = set(news["news_id"].astype(str))
    tweets = tweets[tweets["Id"].astype(str).isin(rated_ids)].copy()
    tweets["news_id"] = tweets["Id"].astype(str)

    meta = users_twitter.drop_duplicates("item_id")
    meta["item_id"] = meta["item_id"].astype(str)

    enriched = news.merge(
        tweets[
            [
                "news_id",
                "QuoteCount",
                "Urls",
                "Mentions",
                "DiversityScore",
                "ContextScore",
                "RecencyScore",
                "SentimentScore",
            ]
        ],
        on="news_id",
        how="left",
    )
    enriched = enriched.merge(
        meta[["item_id", "quote_count"]],
        left_on="news_id",
        right_on="item_id",
        how="left",
        suffixes=("", "_meta"),
    )
    enriched["quote_count"] = enriched["quote_count"].fillna(enriched.get("QuoteCount", 0)).fillna(0)
    enriched["sentiment_score"] = enriched["SentimentScore"].fillna(
        enriched.get("oracle_sentiment_score", 0)
    )
    enriched["diversity_score"] = enriched["DiversityScore"].fillna(0)
    enriched["context_score"] = enriched["ContextScore"].fillna(0)
    enriched["recency_score_oracle"] = enriched["RecencyScore"].fillna(0)
    enriched["mentions_raw"] = enriched["Mentions"].fillna(enriched.get("mentioned_user_ids", ""))
    enriched["urls_raw"] = enriched["Urls"].fillna("")
    enriched["mentions_count"] = enriched["mentions_raw"].apply(lambda x: len(parse_mentions(x)))

    out_dir = Path(cfg.paths.processed_v2_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        "news_id",
        "author_id",
        "text",
        "likes_count",
        "retweets_count",
        "comments_count",
        "quote_count",
        "mentioned_user_ids",
        "mentions_raw",
        "urls_raw",
        "mentions_count",
        "topic",
        "subtopic",
        "created_at",
        "sentiment_score",
        "diversity_score",
        "context_score",
        "recency_score_oracle",
        "oracle_social_capital_score",
    ]
    keep_cols = [c for c in keep_cols if c in enriched.columns]
    out = enriched[keep_cols]
    out.to_csv(out_dir / "news_enriched.csv", index=False)
    pd.read_csv(processed_v1 / "users.csv").to_csv(out_dir / "users.csv", index=False)
    pd.read_csv(processed_v1 / "ratings.csv").to_csv(out_dir / "ratings.csv", index=False)
    return out


def score_components(cfg: V2Config, v1_cfg: AppConfig) -> ComponentFrame:
    news = enrich_news_for_v2(cfg, v1_cfg)
    users = pd.read_csv(Path(cfg.paths.processed_v2_dir) / "users.csv")
    frame = compute_components(
        news,
        users,
        v1_cfg.text,
        cfg.social_capital,
        cfg.recency,
    )
    weights = cfg.social_capital.weights.model_dump()
    frame.df["social_capital_v2"] = frame.df.apply(
        lambda row: weighted_social_capital(
            row, weights, cfg.social_capital.use_extended_formula
        ),
        axis=1,
    )
    frame.df["state_art_score"] = (
        cfg.state_art.content_weight * frame.df["content_relevance"]
        + cfg.state_art.engagement_weight * frame.df["engagement_score"]
    )
    frame.df["b1_score"] = (
        frame.df["likes_count"]
        + frame.df["retweets_count"]
        + frame.df["comments_count"]
        + frame.df["quote_count"]
    )
    interim = Path(cfg.paths.interim_v2_dir)
    interim.mkdir(parents=True, exist_ok=True)
    frame.df.to_csv(interim / "component_scores.csv", index=False)
    return frame
