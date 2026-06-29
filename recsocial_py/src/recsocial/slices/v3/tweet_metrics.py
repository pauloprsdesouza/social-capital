"""V3 tweet metrics and SCSA-PLUS social capital (SDD §14–18)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recsocial.shared.text_preprocess import preprocess_text
from recsocial.shared.config_models import TextConfig
from recsocial.shared.utils import parse_mentions
from recsocial.slices.v3.config import ContextConfig, RecencyV3Config, SocialCapitalV3Config


def recency_score_v3(
    created_at,
    reference_time: datetime | None = None,
    decay_factor: float = 0.1,
) -> float:
    ref = reference_time or datetime.now(timezone.utc)
    ts = pd.to_datetime(created_at, errors="coerce", utc=True)
    if pd.isna(ts):
        return 1.0
    age_seconds = max((ref - ts).total_seconds(), 0)
    return 1.0 / (1.0 + decay_factor * math.log10(1.0 + age_seconds))


def diversity_from_oracle(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def token_count_from_text(text: str) -> int:
    return len(preprocess_text(text or "", TextConfig(min_df=1, remove_stopwords=False)).split())


def context_score_tweet(
    text: str,
    topic_keywords: list[str],
    cfg: ContextConfig,
) -> float:
    if not topic_keywords:
        return 0.0
    docs = [text or ""] + topic_keywords
    vectorizer = TfidfVectorizer(max_features=cfg.max_features)
    matrix = vectorizer.fit_transform(docs)
    if matrix.shape[0] < 2:
        return 0.0
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return float(sims.max()) if len(sims) else 0.0


def mentions_strength(mentioned_ids: list[str], strength_by_user: dict[str, float]) -> float:
    return sum(strength_by_user.get(uid, 0.0) for uid in mentioned_ids)


def scsa_plus_score(
    row: pd.Series,
    author_strength: float,
    mentions_str: float,
    recency: float,
    cfg: SocialCapitalV3Config,
) -> float:
    raw = (
        (author_strength if cfg.include_author_strength else 0.0)
        + float(row.get("retweets_count", 0))
        + float(row.get("likes_count", 0))
        + float(row.get("comments_count", 0))
        + float(row.get("impression_count", 0))
        + int(row.get("media_count", 0))
        + int(row.get("hashtag_count", 0))
        + int(row.get("url_count", 0))
        + float(row.get("diversity_score", 0))
        + (mentions_str if cfg.include_mentions_strength else 0.0)
        + int(row.get("token_count", 0))
        + float(row.get("context_score", 0))
    )
    if cfg.include_recency_multiplier:
        return raw * recency
    return raw


def score_tweets_scsa_plus(
    news_df: pd.DataFrame,
    user_strength_df: pd.DataFrame,
    text_config: TextConfig,
    recency_cfg: RecencyV3Config,
    context_cfg: ContextConfig,
    sc_cfg: SocialCapitalV3Config,
) -> pd.DataFrame:
    df = news_df.copy()
    df["news_id"] = df["news_id"].astype(str)
    strength_map = user_strength_df.set_index("user_id")["user_strength_score"].to_dict()

    ref_time = pd.to_datetime(df["created_at"], errors="coerce", utc=True).max()
    ref_dt = ref_time.to_pydatetime() if pd.notna(ref_time) else datetime.now(timezone.utc)

    topics = [t for t in df.get("topic", pd.Series()).dropna().unique().tolist() if t]
    topic_docs = [str(t) for t in topics] if topics else ["news", "entertainment", "sports"]

    rows = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        mentions = parse_mentions(d.get("mentions_raw", d.get("mentioned_user_ids", "")))
        author = str(d.get("author_id", ""))
        rec = recency_score_v3(d.get("created_at"), ref_dt, recency_cfg.decay_factor)
        ctx = float(d.get("context_score", 0))
        if ctx == 0 and d.get("text"):
            ctx = context_score_tweet(str(d["text"]), topic_docs, context_cfg)
        mstrength = mentions_strength(mentions, strength_map)
        sc = scsa_plus_score(
            pd.Series(d),
            strength_map.get(author, 0.0),
            mstrength,
            rec,
            sc_cfg,
        )
        rows.append(
            {
                "news_id": str(d["news_id"]),
                "author_id": author,
                "recency_score_v3": rec,
                "diversity_score_v3": diversity_from_oracle(d.get("diversity_score", 0)),
                "context_score_v3": ctx,
                "mentions_strength_score": mstrength,
                "author_strength_score": strength_map.get(author, 0.0),
                "token_count": token_count_from_text(str(d.get("text", ""))),
                "scsa_plus_score": sc,
            }
        )
    return pd.DataFrame(rows)
