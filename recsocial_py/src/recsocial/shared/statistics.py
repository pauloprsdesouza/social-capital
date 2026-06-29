"""Shared statistical tests (SDD §23)."""

from __future__ import annotations

import pandas as pd
import scipy.stats as stats

from recsocial.shared.config_models import PairedTestConfig


def run_paired_t_tests(
    metrics_detail: pd.DataFrame,
    cfg: PairedTestConfig,
) -> pd.DataFrame:
    comparisons = cfg.comparisons or []
    metric_cols = ["mrr", "map", "ndcg"]
    rows = []
    for a, b in comparisons:
        da = metrics_detail[metrics_detail["algorithm"] == a].set_index("user_id")
        db = metrics_detail[metrics_detail["algorithm"] == b].set_index("user_id")
        common = da.index.intersection(db.index)
        if len(common) < 2:
            continue
        for metric in metric_cols:
            if metric not in da.columns:
                continue
            x = da.loc[common, metric].astype(float)
            y = db.loc[common, metric].astype(float)
            stat, p_val = stats.ttest_rel(x, y)
            rows.append(
                {
                    "algorithm_a": a,
                    "algorithm_b": b,
                    "metric": metric,
                    "t_statistic": float(stat),
                    "p_value": float(p_val),
                    "significant": float(p_val) <= cfg.significance_level,
                    "n_pairs": len(common),
                }
            )
    return pd.DataFrame(rows)
