"""User profile vectors and incremental updates (SDD §13)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recsocial.slices.v1.config import RecommendationConfig
from recsocial.slices.v1.text_features import FeatureArtifacts, news_id_to_index


class UserProfileStore:
    def __init__(
        self,
        features: FeatureArtifacts,
        config: RecommendationConfig,
    ) -> None:
        self.features = features
        self.config = config
        self._index = news_id_to_index(features.news_ids)
        self._profiles: dict[str, np.ndarray] = {}

    def _rating_weight(self, rating: int) -> float:
        if self.config.profile_weight_mode == "rating_over_5":
            return rating / 5.0
        return float(rating - 3)

    def get_profile(self, user_id: str) -> np.ndarray | None:
        return self._profiles.get(user_id)

    def update_from_ratings(
        self,
        user_id: str,
        rated_news_ids: list[str],
        ratings: list[int],
    ) -> None:
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for nid, rating in zip(rated_news_ids, ratings):
            idx = self._index.get(nid)
            if idx is None:
                continue
            w = self._rating_weight(int(rating))
            if w == 0:
                continue
            vectors.append(self.features.matrix[idx])
            weights.append(w)
        if not vectors:
            return
        w_arr = np.array(weights, dtype=float)
        if np.allclose(w_arr, 0):
            return
        mat = np.vstack(vectors)
        profile = np.average(mat, axis=0, weights=np.abs(w_arr))
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        self._profiles[user_id] = profile

    def similarity(self, user_id: str, news_id: str) -> float:
        profile = self.get_profile(user_id)
        if profile is None:
            return 0.0
        idx = self._index.get(news_id)
        if idx is None:
            return 0.0
        news_vec = self.features.matrix[idx].reshape(1, -1)
        prof = profile.reshape(1, -1)
        return float(cosine_similarity(prof, news_vec)[0, 0])
