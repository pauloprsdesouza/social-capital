"""Publication-style figures for V2 reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.visualization.charts import (
    plot_grouped_metrics_bar,
    plot_precision_at_k,
    plot_ttest_heatmap,
)
from recsocial.shared.visualization.gallery import write_figures_index
from recsocial.slices.v2.config import V2Config

V2_HEADLINE = [
    "B1-STATE_ART",
    "B1-SCSA_PLUS",
    "B1-SCSA_PLUS_V3",
    "SC-STATE_ART",
    "SC-SCSA_PLUS",
    "SC-SCSA_PLUS_V3",
    "SCSA-STATE_ART",
    "SCSA-SCSA_PLUS",
    "SCSA-SCSA_PLUS_V3",
]

V2_PRECISION = [
    "B1-STATE_ART",
    "B1-SCSA_PLUS",
    "SC-STATE_ART",
    "SC-SCSA_PLUS",
    "SCSA-STATE_ART",
    "SCSA-SCSA_PLUS",
]


def generate_v2_figures(
    reports_dir: Path,
    cfg: V2Config,
    ttests: pd.DataFrame | None = None,
) -> dict[str, Path]:
    figures_dir = reports_dir / "figures"
    summary = pd.read_csv(reports_dir / "v2_metrics_summary.csv")
    detail = pd.read_csv(reports_dir / "v2_metrics_detail.csv")

    paths: dict[str, Path] = {
        "metrics_comparison": plot_grouped_metrics_bar(
            summary,
            figures_dir / "fig01_metrics_comparison.png",
            title="V2 — MRR, MAP, NDCG by Algorithm Variant (AMCIS 2024)",
            algorithms=sorted(summary["algorithm"].tolist()),
        ),
        "headline_variants": plot_grouped_metrics_bar(
            summary,
            figures_dir / "fig02_headline_variants.png",
            title="V2 — Headline B1 / SC / SCSA Variants",
            algorithms=[a for a in V2_HEADLINE if a in set(summary["algorithm"])],
        ),
        "precision_at_k": plot_precision_at_k(
            detail,
            figures_dir / "fig03_precision_at_k.png",
            title="V2 — Precision@K (Paper Figure)",
            algorithms=[a for a in V2_PRECISION if a in set(detail["algorithm"])],
        ),
    }

    if ttests is not None and not ttests.empty:
        ttests.to_csv(reports_dir / "paired_ttests.csv", index=False)
        for i, metric in enumerate(("mrr", "map", "ndcg"), start=4):
            paths[f"ttest_{metric}"] = plot_ttest_heatmap(
                ttests,
                figures_dir / f"fig0{i}_ttest_{metric}.png",
                title="V2 — Paired t-tests",
                metric=metric,
            )

    interim = Path(cfg.paths.interim_v2_dir) / "component_scores.csv"
    if interim.exists():
        from recsocial.shared.visualization.charts import plot_distribution

        comp = pd.read_csv(interim)
        if "recency_score" in comp.columns:
            paths["recency_dist"] = plot_distribution(
                comp["recency_score"],
                figures_dir / "fig07_recency_distribution.png",
                title="V2 — Recency Score Distribution",
                xlabel="recency_score",
            )
        if "engagement_score" in comp.columns:
            paths["engagement_dist"] = plot_distribution(
                comp["engagement_score"],
                figures_dir / "fig08_engagement_distribution.png",
                title="V2 — Engagement Score Distribution",
                xlabel="engagement_score",
            )

    gallery = [
        ("fig01_metrics_comparison.png", "All algorithm variants — MRR, MAP, NDCG"),
        ("fig02_headline_variants.png", "Headline B1 / SC / SCSA families"),
        ("fig03_precision_at_k.png", "Precision@1–5 (paper requirement)"),
    ]
    if ttests is not None and not ttests.empty:
        gallery.extend(
            [
                ("fig04_ttest_mrr.png", "Paired t-tests — MRR"),
                ("fig05_ttest_map.png", "Paired t-tests — MAP"),
                ("fig06_ttest_ndcg.png", "Paired t-tests — NDCG"),
            ]
        )
    if "recency_dist" in paths:
        gallery.append(("fig07_recency_distribution.png", "Recency score distribution"))
    if "engagement_dist" in paths:
        gallery.append(("fig08_engagement_distribution.png", "Engagement score distribution"))
    write_figures_index(figures_dir, gallery)
    return paths
