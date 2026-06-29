"""Social Capital scoring (SDD §12, paper Algorithm 3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsocial.slices.v1.config import AppConfig, InfluenceConfig, SentimentConfig
from recsocial.slices.v1.influence import influence_score
from recsocial.shared.schemas import Comment, News, User
from recsocial.slices.v1.sentiment import SentimentAnalyzer, sentiment_weight


@dataclass
class ScoreEngine:
    users: dict[str, User]
    comments_by_news: dict[str, list[Comment]]
    influence_cfg: InfluenceConfig
    sentiment_cfg: SentimentConfig
    sentiment_analyzer: SentimentAnalyzer | None
    max_followers: int
    max_depth: int = 1
    _cache: dict[str, float] = field(default_factory=dict)

    def user_influence(self, user_id: str) -> float:
        key = f"inf:{user_id}"
        if key in self._cache:
            return self._cache[key]
        user = self.users.get(user_id)
        if user is None:
            return 0.0
        val = influence_score(
            user.followers_count,
            user.received_likes_count,
            user.published_news_count,
            user.lists_count,
            user.is_verified,
            mode=self.influence_cfg.mode,
            max_followers=self.max_followers,
            beta_lists_fallback=self.influence_cfg.beta_lists_fallback,
            verified_bonus_theta=self.influence_cfg.verified_bonus_theta,
            lambda_=self.influence_cfg.lambda_,
        )
        self._cache[key] = val
        return val

    def _comment_score(self, comment: Comment, depth: int, visited: set[str]) -> float:
        if depth > self.max_depth or comment.comment_id in visited:
            return 0.0
        visited.add(comment.comment_id)
        author_inf = self.user_influence(comment.author_id)
        base = comment.likes_count + comment.retweets_count
        if self.sentiment_cfg.enabled and self.sentiment_analyzer is not None:
            label = self.sentiment_analyzer.classify(comment.text).label
            return base * author_inf * sentiment_weight(label, self.sentiment_cfg)
        return base * author_inf

    def social_capital_score(
        self,
        news: News,
        *,
        sentiment_enabled: bool = False,
        visited: set[str] | None = None,
    ) -> float:
        cache_key = f"sc:{news.news_id}:{sentiment_enabled}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        visited = visited or set()
        if news.news_id in visited:
            return 0.0
        visited.add(news.news_id)

        stm = sum(self.user_influence(uid) for uid in news.mentioned_user_ids)
        stc = 0.0
        for comment in self.comments_by_news.get(news.news_id, []):
            stc += self._comment_score(comment, depth=1, visited=set())

        base = (
            news.likes_count
            + news.retweets_count
            + stm
            + stc
            + news.comments_count
        )
        author_inf = self.user_influence(news.author_id)
        score = base * author_inf

        if sentiment_enabled and self.sentiment_cfg.enabled and self.sentiment_analyzer:
            label = self.sentiment_analyzer.classify(news.text).label
            score *= sentiment_weight(label, self.sentiment_cfg)

        self._cache[cache_key] = score
        return score


def build_score_engine(
    users_df: pd.DataFrame,
    news_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    cfg: AppConfig,
    sentiment_analyzer: SentimentAnalyzer | None = None,
) -> ScoreEngine:
    from recsocial.slices.v1.data_loader import comments_by_news, news_to_dict, users_to_dict

    users = users_to_dict(users_df)
    news = news_to_dict(news_df)
    comments = comments_by_news(comments_df)
    max_followers = max((u.followers_count for u in users.values()), default=0)
    max_followers = max(max_followers, 1)
    analyzer = sentiment_analyzer
    if analyzer is None and cfg.sentiment.enabled:
        from recsocial.slices.v1.sentiment import build_sentiment_analyzer

        oracle = None
        if "oracle_sentiment_score" in news_df.columns:
            oracle = dict(
                zip(
                    news_df["news_id"].astype(str),
                    news_df["oracle_sentiment_score"].astype(float),
                )
            )
        backend = "oracle" if cfg.sentiment.backend == "oracle" else cfg.sentiment.backend
        analyzer = build_sentiment_analyzer(backend, oracle_scores=oracle)

    return ScoreEngine(
        users=users,
        comments_by_news=comments,
        influence_cfg=cfg.influence,
        sentiment_cfg=cfg.sentiment,
        sentiment_analyzer=analyzer,
        max_followers=max_followers,
    )


def score_all_news(
    engine: ScoreEngine,
    news_df: pd.DataFrame,
    *,
    sentiment_enabled: bool = False,
) -> pd.DataFrame:
    from recsocial.slices.v1.data_loader import news_to_dict

    news = news_to_dict(news_df)
    rows = []
    for nid, item in news.items():
        rows.append(
            {
                "news_id": str(nid),
                "author_id": str(item.author_id),
                "social_capital_score": engine.social_capital_score(
                    item, sentiment_enabled=sentiment_enabled
                ),
                "sentiment_enabled": sentiment_enabled,
            }
        )
    return pd.DataFrame(rows)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_v, max_v = float(scores.min()), float(scores.max())
    if max_v == min_v:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)
