"""Tests for author paper ranking imports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.paper_rankings import load_paper_rankings, normalize_paper_rankings


def test_normalize_paper_rankings_legacy_columns():
    df = pd.DataFrame(
        {
            "user_id": [1.0],
            "algorithm": ["B1-STATE_ART"],
            "item_id": ["1369343506939019269"],
            "ranking": [1],
            "rating": [5.0],
        }
    )
    out = normalize_paper_rankings(df)
    assert list(out.columns) == [
        "user_id",
        "algorithm",
        "news_id",
        "ranking",
        "rating",
        "score",
    ]
    assert out.iloc[0]["news_id"] == "1369343506939019269"
    assert out.iloc[0]["user_id"] == "1"


def test_load_paper_rankings_file_exists():
    path = Path(__file__).resolve().parents[2] / "data/raw/paper_rankings/v2_recommendations.csv"
    if not path.exists():
        return
    df = load_paper_rankings(path)
    assert not df.empty
    assert "B1-STATE_ART" in set(df["algorithm"])
