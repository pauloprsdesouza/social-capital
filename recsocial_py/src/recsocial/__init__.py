"""recsocial — Social Capital recommender reproduction package."""

__version__ = "0.1.0"

from recsocial.slices.v1.config import AppConfig, load_config
from recsocial.shared.evaluation import aggregate_metrics_by_algorithm, mrr, average_precision, ndcg_at_k
from recsocial.slices.v1.experiment import run_experiment
from recsocial.slices.v1.migrate import migrate_to_sdd_schema

__all__ = [
    "AppConfig",
    "load_config",
    "aggregate_metrics_by_algorithm",
    "mrr",
    "average_precision",
    "ndcg_at_k",
    "run_experiment",
    "migrate_to_sdd_schema",
]
