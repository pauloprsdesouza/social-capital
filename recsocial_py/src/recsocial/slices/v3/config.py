"""V3 configuration — SCSA-PLUS (Improving Personalized Recommendations SDD)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from recsocial.shared.config_loader import load_yaml_config, resolve_paths_in_dict
from recsocial.shared.config_models import EvaluationConfig, PairedTestConfig, ReproductionConfig, RerankConfig
from recsocial.slices.v1.config import AppConfig, load_config


class ReputationConfig(BaseModel):
    max_mentions: int = 100
    max_replies: int = 100
    listed_count_theta: float = 0.01
    zero_interaction_strategy: str = "zero"


class InfluenceV3Config(BaseModel):
    max_recent_tweets: int = 100
    include_impressions: bool = True
    log_base: float = 10.0
    zero_followers_strategy: str = "safe_one"


class RecencyV3Config(BaseModel):
    decay_factor: float = 0.1
    unit: str = "seconds"
    formula: str = "logarithmic"


class ContextConfig(BaseModel):
    method: str = "tfidf_cosine"
    use_topic_keywords: bool = True
    max_features: int = 1000


class SocialCapitalV3Config(BaseModel):
    normalize_features: bool = False
    include_author_strength: bool = True
    include_mentions_strength: bool = True
    include_recency_multiplier: bool = True


class PcaConfigV3(BaseModel):
    enabled: bool = True
    variance_ratio: float = 0.95
    max_features: int = 1000
    score_component: int = 0


class V3PathsConfig(BaseModel):
    processed_v3_dir: str = "data/v3/processed"
    interim_v3_dir: str = "data/v3/interim"
    reports_v3_dir: str = "reports/v3"
    v2_config_path: str = "configs/v2.yaml"
    v1_config_path: str = "configs/v1.yaml"


class PaperTargetsV3(BaseModel):
    mrr: dict[str, float] = Field(default_factory=dict)
    map: dict[str, float] = Field(default_factory=dict)
    ndcg: dict[str, float] = Field(default_factory=dict)
    tolerance: float = 0.05
    aliases: dict[str, str] = Field(default_factory=lambda: {"SCSA_PLUS": "SC"})


class V3Config(RerankConfig):
    random_seed: int = 42
    reputation: ReputationConfig = Field(default_factory=ReputationConfig)
    influence: InfluenceV3Config = Field(default_factory=InfluenceV3Config)
    recency: RecencyV3Config = Field(default_factory=RecencyV3Config)
    context: ContextConfig = Field(default_factory=ContextConfig)
    social_capital: SocialCapitalV3Config = Field(default_factory=SocialCapitalV3Config)
    pca: PcaConfigV3 = Field(default_factory=PcaConfigV3)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    statistics: PairedTestConfig = Field(default_factory=PairedTestConfig)
    reproduction: ReproductionConfig = Field(default_factory=ReproductionConfig)
    paths: V3PathsConfig = Field(default_factory=V3PathsConfig)
    paper_targets: PaperTargetsV3 = Field(default_factory=PaperTargetsV3)

    def load_v1(self, base_dir: Path) -> AppConfig:
        return load_config(base_dir / self.paths.v1_config_path, base_dir=base_dir)

    def resolve(self, base_dir: Path) -> V3Config:
        data = self.model_dump()
        data["paths"] = resolve_paths_in_dict(data["paths"], base_dir)
        return V3Config.model_validate(data)


def load_v3_config(path: str | Path, base_dir: Path | None = None) -> V3Config:
    path = Path(path)
    base = base_dir or path.parent.parent
    return load_yaml_config(path, V3Config, base, resolve_fn=lambda c, b: c.resolve(b))
