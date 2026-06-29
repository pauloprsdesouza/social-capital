"""Recommendation algorithms SC, SC+SA, CS-PLUS, B1, HYBRID (SDD §15)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from recsocial.slices.v1.config import AppConfig
from recsocial.shared.schemas import News, RecommendationResult
from recsocial.slices.v1.social_capital import ScoreEngine, normalize_scores
from recsocial.slices.v1.user_profile import UserProfileStore

AlgorithmName = Literal["SC", "SC+SA", "CS-PLUS", "B1", "HYBRID"]

ALGO_MAP = {
    "SC": "SC",
    "SCSA": "SC+SA",
    "SC+SA": "SC+SA",
    "CS": "CS-PLUS",
    "CS-PLUS": "CS-PLUS",
    "B1": "B1",
    "HYBRID": "HYBRID",
}


class Recommender:
    def __init__(
        self,
        cfg: AppConfig,
        engine: ScoreEngine,
        news_df: pd.DataFrame,
        profiles: UserProfileStore,
        sc_scores: pd.DataFrame,
        scsa_scores: pd.DataFrame,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.news_df = news_df.set_index("news_id")
        self.profiles = profiles
        self.sc_scores = sc_scores.set_index("news_id")["social_capital_score"]
        self.scsa_scores = scsa_scores.set_index("news_id")["social_capital_score"]

    def _baseline_score(self, news_id: str) -> float:
        row = self.news_df.loc[news_id]
        return float(row.likes_count + row.retweets_count + row.comments_count)

    def _score_news(
        self,
        user_id: str,
        news_id: str,
        algorithm: AlgorithmName,
    ) -> tuple[float, float | None, float | None]:
        sc: float | None = None
        sim: float | None = None
        if algorithm == "SC":
            sc = float(self.sc_scores.get(news_id, 0.0))
            final = sc
        elif algorithm == "SC+SA":
            sc = float(self.scsa_scores.get(news_id, 0.0))
            final = sc
        elif algorithm == "CS-PLUS":
            sim = self.profiles.similarity(user_id, news_id)
            final = sim
            if self.profiles.get_profile(user_id) is None:
                final = float(self.sc_scores.get(news_id, 0.0))
        elif algorithm == "B1":
            final = self._baseline_score(news_id)
        elif algorithm == "HYBRID":
            sc = float(self.sc_scores.get(news_id, 0.0))
            sim = self.profiles.similarity(user_id, news_id)
            if self.cfg.recommendation.hybrid_mode == "normalized_hybrid":
                sc_norm = sc  # caller normalizes globally when ranking
                final = sc_norm + sim
            else:
                final = sc + (sim if sim > self.cfg.recommendation.similarity_threshold else 0.0)
        else:
            final = 0.0
        return final, sc, sim

    def recommend(
        self,
        user_id: str,
        candidate_news_ids: list[str],
        algorithm: AlgorithmName,
        top_k: int | None = None,
    ) -> list[RecommendationResult]:
        top_k = top_k or self.cfg.recommendation.top_k
        scored: list[tuple[str, float, float | None, float | None]] = []
        for nid in candidate_news_ids:
            if nid not in self.news_df.index:
                continue
            final, sc, sim = self._score_news(user_id, nid, algorithm)
            scored.append((nid, final, sc, sim))

        if algorithm == "HYBRID" and self.cfg.recommendation.hybrid_mode == "normalized_hybrid":
            finals = np.array([s[1] for s in scored], dtype=float)
            scs = np.array([s[2] or 0.0 for s in scored], dtype=float)
            scs_norm = normalize_scores(scs)
            scored = [
                (nid, float(scs_norm[i] + (sim or 0.0)), sc, sim)
                for i, (nid, _, sc, sim) in enumerate(scored)
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[RecommendationResult] = []
        for rank, (nid, final, sc, sim) in enumerate(scored[:top_k], start=1):
            results.append(
                RecommendationResult(
                    user_id=user_id,
                    news_id=nid,
                    algorithm=algorithm,
                    rank=rank,
                    score=final,
                    social_capital_score=sc,
                    similarity_score=sim,
                )
            )
        return results
