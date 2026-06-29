"""Publication-style figures for V1 reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.visualization.charts import (
    plot_grouped_metrics_bar,
    plot_oracle_validation,
    plot_paper_targets,
    plot_precision_at_k,
)
from recsocial.shared.visualization.gallery import write_figures_index
from recsocial.slices.v1.config import AppConfig


V1_ALGORITHMS = ["B1", "CS-PLUS", "SC", "SC+SA"]
PRECISION_BASELINES = ["B1", "CS-PLUS", "SC", "SC+SA"]


def generate_v1_figures(reports_dir: Path, cfg: AppConfig) -> dict[str, Path]:
    figures_dir = reports_dir / "figures"
    summary = pd.read_csv(reports_dir / "trial_metrics_summary.csv")
    detail = pd.read_csv(reports_dir / "trial_metrics_detail.csv")
    oracle = pd.read_csv(reports_dir / "oracle_validation.csv")

    paper_targets = {
        "mrr": cfg.paper_targets.mrr,
        "map": cfg.paper_targets.map,
        "ndcg": cfg.paper_targets.ndcg,
    }

    paths = {
        "metrics_comparison": plot_grouped_metrics_bar(
            summary,
            figures_dir / "fig01_metrics_comparison.png",
            title="V1 — MRR, MAP, NDCG by Algorithm (FedCSIS 2022)",
            algorithms=V1_ALGORITHMS,
            paper_targets=paper_targets,
        ),
        "precision_at_k": plot_precision_at_k(
            detail,
            figures_dir / "fig02_precision_at_k.png",
            title="V1 — Precision@K by Algorithm",
            algorithms=PRECISION_BASELINES,
        ),
        "paper_targets": plot_paper_targets(
            summary,
            paper_targets,
            figures_dir / "fig03_paper_targets.png",
            title="V1 — Measured vs Paper Targets",
            algorithms=V1_ALGORITHMS,
            tolerance=cfg.paper_targets.tolerance,
        ),
        "oracle_validation": plot_oracle_validation(
            oracle,
            figures_dir / "fig04_oracle_validation.png",
            title="V1 — Oracle Social Capital Score Validation",
        ),
    }

    write_figures_index(
        figures_dir,
        [
            ("fig01_metrics_comparison.png", "Algorithm metrics comparison (MRR, MAP, NDCG)"),
            ("fig02_precision_at_k.png", "Precision@1–5 curves"),
            ("fig03_paper_targets.png", "Measured vs paper targets"),
            ("fig04_oracle_validation.png", "Oracle SC score validation"),
        ],
    )
    return paths
