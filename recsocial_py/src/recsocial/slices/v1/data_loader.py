"""Load SDD-schema CSV datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.schemas import Comment, News, Rating, User
from recsocial.shared.utils import safe_str_id


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def load_users(path: Path) -> pd.DataFrame:
    return _read_csv(path)


def load_news(path: Path) -> pd.DataFrame:
    return _read_csv(path)


def load_comments(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "comment_id",
                "news_id",
                "author_id",
                "text",
                "likes_count",
                "retweets_count",
            ]
        )
    return _read_csv(path)


def load_mentions(path: Path) -> pd.DataFrame:
    return _read_csv(path)


def load_ratings(path: Path) -> pd.DataFrame:
    return _read_csv(path)


def load_processed_bundle(processed_dir: Path) -> dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    return {
        "users": load_users(processed_dir / "users.csv"),
        "news": load_news(processed_dir / "news.csv"),
        "comments": load_comments(processed_dir / "comments.csv"),
        "mentions": load_mentions(processed_dir / "mentions.csv"),
        "ratings": load_ratings(processed_dir / "ratings.csv"),
    }


def users_to_dict(df: pd.DataFrame) -> dict[str, User]:
    out: dict[str, User] = {}
    for row in df.itertuples(index=False):
        uid = safe_str_id(row.user_id)
        out[uid] = User(
            user_id=uid,
            followers_count=int(row.followers_count),
            lists_count=int(row.lists_count),
            received_likes_count=int(row.received_likes_count),
            published_news_count=int(row.published_news_count),
            is_verified=bool(row.is_verified),
        )
    return out


def news_to_dict(df: pd.DataFrame) -> dict[str, News]:
    out: dict[str, News] = {}
    for row in df.itertuples(index=False):
        nid = safe_str_id(row.news_id)
        mentions = []
        if hasattr(row, "mentioned_user_ids") and pd.notna(row.mentioned_user_ids):
            mentions = [m for m in str(row.mentioned_user_ids).split("|") if m]
        out[nid] = News(
            news_id=nid,
            author_id=safe_str_id(row.author_id),
            text=str(row.text),
            likes_count=int(row.likes_count),
            retweets_count=int(row.retweets_count),
            comments_count=int(row.comments_count),
            mentioned_user_ids=mentions,
            topic=getattr(row, "topic", None),
            subtopic=getattr(row, "subtopic", None),
        )
    return out


def comments_by_news(df: pd.DataFrame) -> dict[str, list[Comment]]:
    grouped: dict[str, list[Comment]] = {}
    if df.empty:
        return grouped
    for row in df.itertuples(index=False):
        c = Comment(
            comment_id=row.comment_id,
            news_id=row.news_id,
            author_id=row.author_id,
            text=str(row.text),
            likes_count=int(row.likes_count),
            retweets_count=int(row.retweets_count),
        )
        grouped.setdefault(row.news_id, []).append(c)
    return grouped


def ratings_to_models(df: pd.DataFrame) -> list[Rating]:
    models: list[Rating] = []
    for row in df.itertuples(index=False):
        models.append(
            Rating(
                user_id=safe_str_id(row.user_id),
                news_id=safe_str_id(row.news_id),
                round_id=int(row.round_id),
                position=int(row.position),
                algorithm=str(row.algorithm),
                rating=int(row.rating),
            )
        )
    return models
