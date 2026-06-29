"""Compare experiment metrics against legacy CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.evaluation import aggregate_metrics_by_algorithm


def _normalize_legacy_recommendations(legacy: pd.DataFrame) -> pd.DataFrame:
    df = legacy.copy()
    if "item_id" in df.columns and "news_id" not in df.columns:
        df = df.rename(columns={"item_id": "news_id"})
    if "news_id" in df.columns:
        df["news_id"] = df["news_id"].astype(str)
    return df


def compare_summary_to_legacy(
    summary: pd.DataFrame,
    legacy_path: Path,
    *,
    version_prefix: str,
    algorithms: list[str] | None = None,
) -> pd.DataFrame:
    """Join current summary with legacy aggregate metrics per algorithm."""
    legacy = _normalize_legacy_recommendations(pd.read_csv(legacy_path))
    legacy_metrics = aggregate_metrics_by_algorithm(
        legacy,
        user_col="user_id",
        algo_col="algorithm",
        rank_col="ranking",
        rating_col="rating",
    )

    algo_list = algorithms or sorted(
        set(summary["algorithm"]) | set(legacy_metrics.keys())
    )
    rows = []
    for algo in algo_list:
        cur = summary[summary["algorithm"] == algo]
        if cur.empty:
            continue
        leg = legacy_metrics.get(algo, {})
        rows.append(
            {
                "algorithm": algo,
                f"mrr_{version_prefix}": float(cur["mrr"].iloc[0]),
                "mrr_legacy": leg.get("mrr"),
                f"map_{version_prefix}": float(cur["map"].iloc[0]),
                "map_legacy": leg.get("map"),
                f"ndcg_{version_prefix}": float(cur["ndcg"].iloc[0]),
                "ndcg_legacy": leg.get("ndcg"),
            }
        )
    return pd.DataFrame(rows)
