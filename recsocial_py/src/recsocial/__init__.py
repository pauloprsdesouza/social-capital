"""recsocial — Social Capital recommender reproduction (V1 / V2 / V3).

Quick start::

    recsocial run all      # full pipeline + validation
    recsocial validate     # compare reports to paper targets

Documentation: see ``docs/README.md`` in the repository root.

Public API:
    - V1: ``recsocial.slices.v1`` (``run_experiment``, ``load_config``)
    - V2: ``recsocial.slices.v2`` (``run_v2_experiment``, ``load_v2_config``)
    - V3: ``recsocial.slices.v3`` (``run_v3_experiment``, ``load_v3_config``)
    - Pipelines: ``recsocial.shared.pipeline``
    - Validation: ``recsocial.shared.paper_validation``
"""

__version__ = "0.2.0"

from recsocial.shared.evaluation import aggregate_metrics_by_algorithm, average_precision, mrr, ndcg_at_k
from recsocial.shared.pipeline import package_root, run_all_pipelines, run_v1_pipeline, run_v2_pipeline, run_v3_pipeline
from recsocial.slices.v1.config import AppConfig, load_config
from recsocial.slices.v1.experiment import run_experiment
from recsocial.slices.v1.migrate import migrate_to_sdd_schema

__all__ = [
    "__version__",
    "AppConfig",
    "load_config",
    "run_experiment",
    "migrate_to_sdd_schema",
    "run_v1_pipeline",
    "run_v2_pipeline",
    "run_v3_pipeline",
    "run_all_pipelines",
    "package_root",
    "aggregate_metrics_by_algorithm",
    "mrr",
    "average_precision",
    "ndcg_at_k",
]
