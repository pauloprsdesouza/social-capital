"""Cross-slice configuration models (DIP: shared does not depend on slices)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextConfig(BaseModel):
    language: str = "en"
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags_symbol: bool = True
    remove_stopwords: bool = True
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["mrr", "precision_at_k", "map", "ndcg"])
    precision_k_values: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    ndcg_k: int = 10
    graded_ndcg: bool = True
    relevance_threshold: int = 4
    # session_list: enumerate sorted ratings (FedCSIS 2022 trial replay)
    # paper_notebook: rank-column metrics (AMCIS/V3 papers, legacy notebook)
    metric_protocol: str = "session_list"
    # sdd: AP / relevant@k; fedcsis_pooled: MAP for multi-item rank slots
    map_protocol: str = "sdd"


class RerankConfig(BaseModel):
    """Shared re-ranking trial structure used by V2 and V3."""

    base_algorithms: list[str] = Field(default_factory=lambda: ["B1", "CS", "SC", "SCSA"])
    rerank_suffixes: dict[str, str] = Field(
        default_factory=lambda: {
            "state_art": "STATE_ART",
            "scsa_plus": "SCSA_PLUS",
            "scsa_plus_v3": "SCSA_PLUS_V3",
        }
    )
