"""V2 component scores: SI, ES, CR, NI, AI, CV, recency, context (SDD §9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from recsocial.shared.config_models import TextConfig
from recsocial.shared.parsing import collection_len
from recsocial.shared.text_preprocess import preprocess_text
from recsocial.shared.utils import parse_mentions
from recsocial.slices.v2.config import RecencyConfig, SocialCapitalV2Config


@dataclass
class ComponentFrame:
    df: pd.DataFrame
    tfidf_vectorizer: TfidfVectorizer | None = None


def _count_field(value) -> int:
    return len(parse_mentions(value))


def _parse_urls(value) -> int:
    count = collection_len(value)
    if count:
        return count
    text = str(value or "").strip()
    return 1 if "http" in text else 0


def _scale_series(
    series: pd.Series,
    *,
    mode: str,
    normalize: bool,
) -> pd.Series:
    if not normalize:
        return series.astype(float)
    values = series.astype(float).values.reshape(-1, 1)
    if mode == "minmax_0_1":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    return pd.Series(scaler.fit_transform(values).flatten(), index=series.index)


def sentiment_impact_from_compound(compound: float) -> float:
    return float((compound + 1.0) / 2.0)


def build_content_relevance(
    texts: pd.Series,
    text_config: TextConfig,
) -> tuple[pd.Series, TfidfVectorizer]:
    cleaned = texts.fillna("").apply(lambda t: preprocess_text(str(t), text_config))
    vectorizer = TfidfVectorizer(
        min_df=max(1, text_config.min_df),
        max_df=text_config.max_df,
        ngram_range=text_config.ngram_range,
    )
    matrix = vectorizer.fit_transform(cleaned)
    relevance = pd.Series(matrix.mean(axis=1).A1, index=texts.index)
    return relevance, vectorizer


def recency_score_series(
    created_at: pd.Series,
    config: RecencyConfig,
) -> pd.Series:
    dates = pd.to_datetime(created_at, errors="coerce", utc=True)
    max_date = dates.max()
    if pd.isna(max_date):
        return pd.Series(1.0, index=created_at.index)
    elapsed_days = (max_date - dates).dt.total_seconds() / 86400.0
    elapsed_days = elapsed_days.fillna(0).clip(lower=0)
    return np.exp(-config.lambda_decay * elapsed_days)


def compute_components(
    news_df: pd.DataFrame,
    users_df: pd.DataFrame,
    text_config: TextConfig,
    sc_config: SocialCapitalV2Config,
    recency_config: RecencyConfig,
) -> ComponentFrame:
    df = news_df.copy()
    df["news_id"] = df["news_id"].astype(str)

    if "quote_count" not in df.columns:
        df["quote_count"] = 0
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    if "diversity_score" not in df.columns:
        df["diversity_score"] = 0.0
    if "context_score" not in df.columns:
        df["context_score"] = 0.0

    df["mentions_count"] = df.get("mentions_raw", df.get("mentioned_user_ids", "")).apply(
        _count_field
    )
    df["urls_count"] = df.get("urls_raw", "").apply(_parse_urls)

    interaction_cols = ["likes_count", "retweets_count", "comments_count", "quote_count"]
    for col in interaction_cols:
        if col not in df.columns:
            df[col] = 0

    scaled_interactions = {}
    for col in interaction_cols:
        scaled_interactions[col] = _scale_series(
            df[col],
            mode=sc_config.scaling_mode,
            normalize=sc_config.normalize_components,
        )

    df["sentiment_impact"] = _scale_series(
        df["sentiment_score"],
        mode=sc_config.scaling_mode,
        normalize=sc_config.normalize_components,
    )
    df["engagement_score"] = pd.concat(scaled_interactions.values(), axis=1).mean(axis=1)
    df["content_relevance"], vectorizer = build_content_relevance(df["text"], text_config)
    df["network_influence"] = df["mentions_count"] + df["urls_count"]
    if sc_config.normalize_components and sc_config.scaling_mode == "minmax_0_1":
        df["network_influence"] = _scale_series(
            df["network_influence"], mode="minmax_0_1", normalize=True
        )

    if sc_config.author_influence_mode == "user_strength":
        users = users_df.set_index("user_id")
        df["author_influence"] = df["author_id"].astype(str).map(
            lambda a: users.loc[a, "followers_count"] if a in users.index else 0
        )
        df["author_influence"] = _scale_series(
            df["author_influence"],
            mode=sc_config.scaling_mode,
            normalize=sc_config.normalize_components,
        )
    else:
        df["author_influence"] = df["mentions_count"].astype(float)

    df["content_virality"] = (
        scaled_interactions["retweets_count"] + scaled_interactions["quote_count"]
    )
    if recency_config.enabled:
        df["recency_score"] = recency_score_series(df.get("created_at", pd.Series()), recency_config)
    else:
        df["recency_score"] = 0.0
    df["diversity_score"] = df["diversity_score"].astype(float)
    df["context_score"] = df["context_score"].astype(float)

    return ComponentFrame(df=df, tfidf_vectorizer=vectorizer)


def weighted_social_capital(row: pd.Series, weights: dict[str, float], extended: bool) -> float:
    base = (
        weights["sentiment_impact"] * row["sentiment_impact"]
        + weights["engagement_score"] * row["engagement_score"]
        + weights["content_relevance"] * row["content_relevance"]
        + weights["network_influence"] * row["network_influence"]
        + weights["author_influence"] * row["author_influence"]
        + weights["content_virality"] * row["content_virality"]
    )
    if extended:
        base += (
            weights.get("recency_score", 0.0) * row.get("recency_score", 0.0)
            + weights.get("diversity_score", 0.0) * row.get("diversity_score", 0.0)
            + weights.get("context_score", 0.0) * row.get("context_score", 0.0)
        )
    return float(base)
