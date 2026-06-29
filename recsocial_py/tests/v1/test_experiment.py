"""Integration tests with synthetic CSV fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsocial.slices.v1.config import AppConfig, PcaConfig, TextConfig
from recsocial.slices.v1.experiment import evaluate_trial_ratings
from recsocial.slices.v1.recommenders import Recommender
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
from recsocial.slices.v1.text_features import build_hybrid_feature_matrix
from recsocial.slices.v1.user_profile import UserProfileStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def mini_bundle() -> dict[str, pd.DataFrame]:
    return {
        "users": pd.read_csv(FIXTURES / "users.csv"),
        "news": pd.read_csv(FIXTURES / "news.csv"),
        "comments": pd.read_csv(FIXTURES / "comments.csv"),
        "mentions": pd.read_csv(FIXTURES / "mentions.csv"),
        "ratings": pd.read_csv(FIXTURES / "ratings.csv"),
    }


@pytest.fixture
def mini_cfg() -> AppConfig:
    return AppConfig(
        text=TextConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
        pca=PcaConfig(enabled=False),
    )


def test_sc_ranks_high_interaction_news_higher(mini_bundle, mini_cfg):
    engine = build_score_engine(
        mini_bundle["users"],
        mini_bundle["news"],
        mini_bundle["comments"],
        mini_cfg,
    )
    scores = score_all_news(engine, mini_bundle["news"], sentiment_enabled=False)
    ranked = scores.sort_values("social_capital_score", ascending=False)
    assert ranked.iloc[0]["news_id"] == "n3"


def test_recommender_produces_top_k(mini_bundle, mini_cfg):
    engine = build_score_engine(
        mini_bundle["users"],
        mini_bundle["news"],
        mini_bundle["comments"],
        mini_cfg,
    )
    sc = score_all_news(engine, mini_bundle["news"], sentiment_enabled=False)
    scsa = score_all_news(engine, mini_bundle["news"], sentiment_enabled=True)
    features = build_hybrid_feature_matrix(
        mini_bundle["news"], mini_bundle["users"], mini_cfg.text, mini_cfg.pca
    )
    profiles = UserProfileStore(features, mini_cfg.recommendation)
    rec = Recommender(mini_cfg, engine, mini_bundle["news"], profiles, sc, scsa)
    out = rec.recommend("u1", ["n1", "n2", "n3"], "SC", top_k=2)
    assert len(out) == 2
    assert out[0].rank == 1


def test_trial_metrics_runs(mini_bundle, mini_cfg):
    metrics = evaluate_trial_ratings(mini_cfg, mini_bundle["ratings"])
    assert not metrics.empty
    assert "mrr" in metrics.columns
