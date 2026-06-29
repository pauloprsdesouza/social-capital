"""V2 configuration (Enhanced Social Capital SDD §8)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from recsocial.shared.config_loader import load_yaml_config, resolve_paths_in_dict
from recsocial.shared.config_models import EvaluationConfig, RerankConfig
from recsocial.slices.v1.config import AppConfig, PathsConfig, load_config


class SocialCapitalWeights(BaseModel):
    sentiment_impact: float = 0.20
    engagement_score: float = 0.20
    content_relevance: float = 0.20
    network_influence: float = 0.20
    author_influence: float = 0.10
    content_virality: float = 0.10
    recency_score: float = 0.0
    diversity_score: float = 0.0
    context_score: float = 0.0


class RecencyConfig(BaseModel):
    enabled: bool = True
    lambda_decay: float = 0.03
    unit: str = "days"


class SocialCapitalV2Config(BaseModel):
    normalize_components: bool = True
    scaling_mode: str = "legacy_notebook"
    author_influence_mode: str = "paper_compatible"
    weights: SocialCapitalWeights = Field(default_factory=SocialCapitalWeights)
    use_extended_formula: bool = False


class StateArtConfig(BaseModel):
    mode: str = "content_plus_engagement"
    content_weight: float = 1.0
    engagement_weight: float = 1.0


class V2PathsConfig(PathsConfig):
    processed_v2_dir: str = "data/v2/processed"
    interim_v2_dir: str = "data/v2/interim"
    reports_v2_dir: str = "reports/v2"
    legacy_recommendations: str = "data/raw/legacy/v2_recommendations.csv"


class StatisticsConfigV2(BaseModel):
    paired_t_test: bool = True
    significance_level: float = 0.05
    comparisons: list[list[str]] = Field(default_factory=list)


class V2Config(RerankConfig):
    v1_config_path: str = "configs/v1.yaml"
    random_seed: int = 42
    recency: RecencyConfig = Field(default_factory=RecencyConfig)
    social_capital: SocialCapitalV2Config = Field(default_factory=SocialCapitalV2Config)
    state_art: StateArtConfig = Field(default_factory=StateArtConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    statistics: StatisticsConfigV2 = Field(default_factory=StatisticsConfigV2)
    paths: V2PathsConfig = Field(default_factory=V2PathsConfig)
    trial_algo_map: dict[str, str] = Field(
        default_factory=lambda: {"SC": "SC", "SCSA": "SCSA", "CS": "CS", "B1": "B1"}
    )

    def load_v1(self, base_dir: Path) -> AppConfig:
        return load_config(base_dir / self.v1_config_path, base_dir=base_dir)

    def resolve(self, base_dir: Path) -> V2Config:
        data = self.model_dump()
        data["paths"] = resolve_paths_in_dict(data["paths"], base_dir)
        return V2Config.model_validate(data)


def load_v2_config(path: str | Path, base_dir: Path | None = None) -> V2Config:
    path = Path(path)
    base = base_dir or path.parent.parent
    return load_yaml_config(path, V2Config, base, resolve_fn=lambda c, b: c.resolve(b))
