"""V3 feature pipeline — merge V2 components, SCSA-PLUS, PCA scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.parsing import collection_len
from recsocial.slices.v2.config import load_v2_config
from recsocial.slices.v2.features import enrich_news_for_v2, score_components
from recsocial.slices.v3.config import V3Config
from recsocial.slices.v3.pca_ranking import compute_pca_scores
from recsocial.slices.v3.tweet_metrics import score_tweets_scsa_plus
from recsocial.slices.v3.user_metrics import build_user_strength_table


def build_v3_features(cfg: V3Config, package_root: Path) -> pd.DataFrame:
    v1_cfg = cfg.load_v1(package_root)
    v2_cfg = load_v2_config(package_root / cfg.paths.v2_config_path, base_dir=package_root)

    enriched = enrich_news_for_v2(v2_cfg, v1_cfg)
    v2_frame = score_components(v2_cfg, v1_cfg)
    users = pd.read_csv(Path(v2_cfg.paths.processed_v2_dir) / "users.csv")

    raw_tweets = pd.read_csv(v1_cfg.paths.raw_tweets)
    raw_tweets["news_id"] = raw_tweets["Id"].astype(str)
    enriched = enriched.merge(
        raw_tweets[["news_id", "Tokens", "Hashtags", "ImpressionCount"]].rename(
            columns={
                "Tokens": "tokens_raw",
                "Hashtags": "hashtags_raw",
                "ImpressionCount": "impression_count",
            }
        ),
        on="news_id",
        how="left",
    )
    enriched["impression_count"] = enriched["impression_count"].fillna(0).astype(int)
    enriched["hashtag_count"] = enriched["hashtags_raw"].apply(collection_len)
    enriched["url_count"] = enriched["urls_raw"].apply(collection_len)
    enriched["media_count"] = 0

    user_strength = build_user_strength_table(
        users, enriched, cfg.reputation, cfg.influence
    )
    scsa = score_tweets_scsa_plus(
        enriched,
        user_strength,
        v1_cfg.text,
        cfg.recency,
        cfg.context,
        cfg.social_capital,
    )
    scsa_cols = [c for c in scsa.columns if c not in {"author_id"}]

    df = v2_frame.df.merge(scsa[scsa_cols], on="news_id", how="left")
    if "author_id" not in df.columns:
        df = df.merge(
            enriched[["news_id", "author_id"]],
            on="news_id",
            how="left",
        )
    df["author_id"] = df["author_id"].astype(str)
    df = df.merge(
        user_strength,
        left_on="author_id",
        right_on="user_id",
        how="left",
        suffixes=("", "_us"),
    )
    df = compute_pca_scores(df, cfg.pca)

    out_dir = Path(cfg.paths.processed_v3_dir)
    interim = Path(cfg.paths.interim_v3_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim / "v3_feature_scores.csv", index=False)
    user_strength.to_csv(out_dir / "user_strength.csv", index=False)
    pd.read_csv(Path(v2_cfg.paths.processed_v2_dir) / "ratings.csv").to_csv(
        out_dir / "ratings.csv", index=False
    )
    return df
