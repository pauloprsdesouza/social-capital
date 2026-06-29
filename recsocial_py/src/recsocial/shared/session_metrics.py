"""Per-session recommendation evaluation (MRR, MAP, NDCG, Precision@K)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recsocial.shared.config_models import EvaluationConfig
from recsocial.shared.evaluation import average_precision, mrr, ndcg_at_k, precision_at_k
from recsocial.shared.paper_metrics import evaluate_session_paper_notebook, map_fedcsis_pooled


@dataclass(frozen=True)
class EvaluationSettings:
    relevance_threshold: int = 4
    ndcg_k: int = 10
    graded_ndcg: bool = True
    precision_k_values: tuple[int, ...] = (1, 2, 3, 4, 5, 10)
    rank_col: str = "ranking"
    algorithm_col: str = "algorithm"
    user_col: str = "user_id"
    rating_col: str = "rating"
    algorithm_aliases: dict[str, str] | None = None
    metric_protocol: str = "session_list"
    map_protocol: str = "sdd"


def settings_from_evaluation_config(
    cfg: EvaluationConfig,
    *,
    rank_col: str = "ranking",
    algorithm_aliases: dict[str, str] | None = None,
) -> EvaluationSettings:
    return EvaluationSettings(
        relevance_threshold=cfg.relevance_threshold,
        ndcg_k=cfg.ndcg_k,
        graded_ndcg=cfg.graded_ndcg,
        precision_k_values=tuple(cfg.precision_k_values),
        rank_col=rank_col,
        algorithm_aliases=algorithm_aliases,
        metric_protocol=getattr(cfg, "metric_protocol", "session_list"),
        map_protocol=getattr(cfg, "map_protocol", "sdd"),
    )


def summarize_session_metrics(metrics_detail: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_detail.groupby("algorithm")[["mrr", "map", "ndcg"]]
        .mean(numeric_only=True)
        .reset_index()
    )


def evaluate_recommendations_by_session(
    recommendations: pd.DataFrame,
    settings: EvaluationSettings,
) -> pd.DataFrame:
    df = recommendations.copy()
    if settings.algorithm_aliases:
        df[settings.algorithm_col] = df[settings.algorithm_col].map(
            lambda a: settings.algorithm_aliases.get(a, a)
        )

    rows: list[dict] = []
    group_cols = [settings.user_col, settings.algorithm_col]
    for (_, algo), grp in df.groupby(group_cols, sort=False):
        if settings.metric_protocol == "paper_notebook":
            row = evaluate_session_paper_notebook(grp, settings)
            row[settings.user_col] = grp[settings.user_col].iloc[0]
            row[settings.algorithm_col] = algo
            rows.append(row)
            continue

        rlist = grp.sort_values(settings.rank_col)[settings.rating_col].tolist()
        map_fn = (
            map_fedcsis_pooled
            if settings.map_protocol == "fedcsis_pooled"
            else lambda r, k, t: average_precision(r, k=k, threshold=t)
        )
        row = {
            settings.user_col: grp[settings.user_col].iloc[0],
            settings.algorithm_col: algo,
            "mrr": mrr(rlist, settings.relevance_threshold),
            "map": map_fn(rlist, settings.ndcg_k, settings.relevance_threshold),
            "ndcg": ndcg_at_k(
                rlist, k=settings.ndcg_k, graded=settings.graded_ndcg
            ),
        }
        for pk in settings.precision_k_values:
            row[f"precision_at_{pk}"] = precision_at_k(
                rlist, pk, settings.relevance_threshold
            )
        rows.append(row)
    return pd.DataFrame(rows)
