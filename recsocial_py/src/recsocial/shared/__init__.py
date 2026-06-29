"""Cross-cutting utilities shared by V1/V2/V3 slices."""

from recsocial.shared.config_models import EvaluationConfig, TextConfig
from recsocial.shared.evaluation import (
    aggregate_metrics_by_algorithm,
    average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
)
from recsocial.shared.reranking import rerank_by_score
from recsocial.shared.session_metrics import evaluate_recommendations_by_session

__all__ = [
    "TextConfig",
    "EvaluationConfig",
    "aggregate_metrics_by_algorithm",
    "average_precision",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "rerank_by_score",
    "evaluate_recommendations_by_session",
]
