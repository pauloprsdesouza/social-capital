"""TF-IDF and hybrid feature matrices (SDD §9.3–9.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsocial.shared.config_models import TextConfig
from recsocial.slices.v1.config import PcaConfig
from recsocial.shared.text_preprocess import preprocess_text


@dataclass
class FeatureArtifacts:
    news_ids: list[str]
    matrix: np.ndarray
    vectorizer: TfidfVectorizer | None
    reducer: TruncatedSVD | None
    feature_names: list[str]


NUMERIC_COLS = [
    "likes_count",
    "retweets_count",
    "comments_count",
    "author_followers",
    "author_lists",
    "author_published_news",
    "mention_count",
]


def _enrich_news(news_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    users = users_df.set_index("user_id")
    df = news_df.copy()
    df["mention_count"] = df["mentioned_user_ids"].fillna("").apply(
        lambda x: len([p for p in str(x).split("|") if p])
    )
    author_cols = users.rename(
        columns={
            "followers_count": "author_followers",
            "lists_count": "author_lists",
            "published_news_count": "author_published_news",
        }
    )
    df = df.merge(
        author_cols[["author_followers", "author_lists", "author_published_news"]],
        left_on="author_id",
        right_index=True,
        how="left",
    )
    return df.fillna(0)


def build_hybrid_feature_matrix(
    news_df: pd.DataFrame,
    users_df: pd.DataFrame,
    text_config: TextConfig,
    pca_config: PcaConfig,
) -> FeatureArtifacts:
    df = _enrich_news(news_df, users_df)
    news_ids = df["news_id"].astype(str).tolist()
    cleaned = df["text"].fillna("").apply(lambda t: preprocess_text(str(t), text_config))

    vectorizer = TfidfVectorizer(
        min_df=text_config.min_df,
        max_df=text_config.max_df,
        ngram_range=text_config.ngram_range,
    )
    tfidf = vectorizer.fit_transform(cleaned)
    numeric = df[NUMERIC_COLS].astype(float).values
    numeric = normalize(numeric, norm="l2", axis=1)
    hybrid = np.hstack([tfidf.toarray(), numeric])

    reducer = None
    if pca_config.enabled:
        n_comp = min(pca_config.n_components, hybrid.shape[1] - 1, hybrid.shape[0] - 1)
        n_comp = max(2, n_comp)
        reducer = TruncatedSVD(n_components=n_comp, random_state=pca_config.random_state)
        matrix = reducer.fit_transform(hybrid)
    else:
        matrix = hybrid

    matrix = normalize(matrix, norm="l2", axis=1)
    names = list(vectorizer.get_feature_names_out()) + NUMERIC_COLS
    return FeatureArtifacts(
        news_ids=news_ids,
        matrix=matrix,
        vectorizer=vectorizer,
        reducer=reducer,
        feature_names=names,
    )


def news_id_to_index(news_ids: list[str]) -> dict[str, int]:
    return {nid: i for i, nid in enumerate(news_ids)}
