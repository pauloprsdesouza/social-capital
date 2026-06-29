"""Social capital scoring tests."""

from recsocial.slices.v1.config import AppConfig, PcaConfig, TextConfig
from recsocial.slices.v1.influence import influence_score, popularity_score
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
import pandas as pd
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _mini_cfg() -> AppConfig:
    return AppConfig(
        text=TextConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
        pca=PcaConfig(enabled=False),
    )


def test_popularity_monotonic():
    assert popularity_score(0) < popularity_score(100)


def test_influence_strict_mode():
    s = influence_score(100, 50, 10, 0, False, mode="strict_equation", max_followers=1000)
    assert s > 0


def test_social_capital_positive():
    users = pd.read_csv(FIXTURES / "users.csv")
    news = pd.read_csv(FIXTURES / "news.csv")
    comments = pd.read_csv(FIXTURES / "comments.csv")
    engine = build_score_engine(users, news, comments, _mini_cfg())
    scores = score_all_news(engine, news, sentiment_enabled=False)
    assert (scores["social_capital_score"] > 0).all()
