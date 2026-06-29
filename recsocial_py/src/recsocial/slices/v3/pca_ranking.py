"""PCA ranking feature pipeline (MainV3.ipynb)."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from recsocial.shared.parsing import parse_collection
from recsocial.slices.v3.config import PcaConfigV3


def combined_text(row: pd.Series) -> str:
    tokens = " ".join(parse_collection(row.get("tokens_raw", row.get("Tokens", ""))))
    hashtags = " ".join(parse_collection(row.get("hashtags_raw", row.get("Hashtags", ""))))
    text = str(row.get("text", ""))
    return f"{text} {tokens} {hashtags}".strip()


def compute_pca_scores(
    feature_df: pd.DataFrame,
    cfg: PcaConfigV3,
) -> pd.DataFrame:
    """Build hybrid matrix (metrics + TF-IDF) and extract PCA score."""
    df = feature_df.copy()
    df["combined_text"] = df.apply(combined_text, axis=1)

    metric_cols = [
        c
        for c in [
            "sentiment_score",
            "diversity_score",
            "context_score",
            "likes_count",
            "comments_count",
            "quote_count",
            "retweets_count",
            "impression_count",
            "scsa_plus_score",
            "social_capital_v2",
        ]
        if c in df.columns
    ]
    metrics = df[metric_cols].fillna(0).astype(float)

    vectorizer = TfidfVectorizer(max_features=cfg.max_features)
    tfidf = vectorizer.fit_transform(df["combined_text"].fillna(""))
    tfidf_df = pd.DataFrame(
        tfidf.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf.shape[1])],
    )

    combined = pd.concat([metrics.reset_index(drop=True), tfidf_df], axis=1)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(combined)

    if not cfg.enabled:
        df["pca1_score"] = combined.sum(axis=1)
        return df

    pca = PCA(n_components=cfg.variance_ratio, random_state=42)
    pca_matrix = pca.fit_transform(normalized)
    comp_idx = cfg.score_component
    df["pca1_score"] = pca_matrix[:, comp_idx]
    df["pca_variance_explained"] = pca.explained_variance_ratio_.sum()
    df["pca_n_components"] = pca_matrix.shape[1]
    return df
