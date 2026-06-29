"""Pydantic domain models (SDD §6)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class User(BaseModel):
    user_id: str
    followers_count: int = 0
    lists_count: int = 0
    received_likes_count: int = 0
    published_news_count: int = 0
    is_verified: bool = False


class News(BaseModel):
    news_id: str
    author_id: str
    text: str
    likes_count: int = 0
    retweets_count: int = 0
    comments_count: int = 0
    mentioned_user_ids: list[str] = Field(default_factory=list)
    comment_ids: list[str] = Field(default_factory=list)
    topic: str | None = None
    subtopic: str | None = None
    created_at: datetime | None = None


class Comment(BaseModel):
    comment_id: str
    news_id: str
    author_id: str
    text: str
    likes_count: int = 0
    retweets_count: int = 0
    mentioned_user_ids: list[str] = Field(default_factory=list)


class Rating(BaseModel):
    user_id: str
    news_id: str
    round_id: int
    position: int
    algorithm: str
    rating: int


class RecommendationResult(BaseModel):
    user_id: str
    news_id: str
    algorithm: str
    rank: int
    score: float
    social_capital_score: float | None = None
    similarity_score: float | None = None
    sentiment_label: str | None = None
    sentiment_weight: float | None = None
