"""V1 configuration — paper baseline (FedCSIS 2022)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recsocial.shared.config_loader import load_yaml_config, resolve_paths_in_dict
from recsocial.shared.config_models import EvaluationConfig, TextConfig

# Re-export for backward compatibility
__all__ = [
    "AppConfig",
    "EvaluationConfig",
    "TextConfig",
    "load_config",
    "InfluenceConfig",
    "SentimentConfig",
    "RecommendationConfig",
    "PathsConfig",
    "PaperTargets",
]


class InfluenceConfig(BaseModel):
    mode: str = "paper_pseudocode"
    lambda_: float = 1.0
    beta_lists_fallback: float = 1.0
    verified_bonus_theta: float = 1.0
    zero_followers_strategy: str = "max_followers"


class SentimentWeights(BaseModel):
    positive: float = 1.5
    negative: float = 1.0
    neutral: float = 0.5
    mixed: float = 0.5
    unknown: float = 1.0


class SentimentConfig(BaseModel):
    enabled: bool = True
    backend: str = "vader"
    weights: SentimentWeights = Field(default_factory=SentimentWeights)


class PcaConfig(BaseModel):
    enabled: bool = True
    n_components: int = 100
    random_state: int = 42


class RecommendationConfig(BaseModel):
    relevance_threshold: int = 4
    similarity_threshold: float = 0.7
    top_k: int = 10
    rounds: int = 4
    profile_weight_mode: str = "rating_minus_3"
    hybrid_mode: str = "raw_hybrid"


class PathsConfig(BaseModel):
    raw_ratings: str = "data/raw/ratings.csv"
    raw_tweets: str = "data/raw/tweets.csv"
    raw_users_twitter: str = "data/raw/users_twitter.csv"
    processed_dir: str = "data/v1/processed"
    interim_dir: str = "data/v1/interim"
    reports_dir: str = "reports/v1"


class PaperTargets(BaseModel):
    mrr: dict[str, float] = Field(default_factory=dict)
    map: dict[str, float] = Field(default_factory=dict)
    ndcg: dict[str, float] = Field(default_factory=dict)
    tolerance: float = 0.05


class AppConfig(BaseModel):
    random_seed: int = 42
    influence: InfluenceConfig = Field(default_factory=InfluenceConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    pca: PcaConfig = Field(default_factory=PcaConfig)
    recommendation: RecommendationConfig = Field(default_factory=RecommendationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    paper_targets: PaperTargets = Field(default_factory=PaperTargets)
    algorithm_aliases: dict[str, str] = Field(default_factory=dict)

    def resolve(self, base_dir: Path) -> AppConfig:
        data = self.model_dump()
        data["paths"] = resolve_paths_in_dict(
            data["paths"],
            base_dir,
            ("raw_ratings", "raw_tweets", "raw_users_twitter", "processed_dir", "interim_dir", "reports_dir"),
        )
        if isinstance(data["text"]["ngram_range"], list):
            data["text"]["ngram_range"] = tuple(data["text"]["ngram_range"])
        return AppConfig.model_validate(data)


def load_config(path: str | Path, base_dir: Path | None = None) -> AppConfig:
    path = Path(path)
    base = base_dir or path.parent.parent
    return load_yaml_config(path, AppConfig, base, resolve_fn=lambda c, b: c.resolve(b))
