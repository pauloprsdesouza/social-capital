"""Publication-style figures for V3 reproduction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.visualization.charts import (
    plot_correlation_heatmap,
    plot_grouped_metrics_bar,
    plot_legacy_comparison,
    plot_precision_at_k,
    plot_ranking_shifts,
    plot_ttest_heatmap,
)
from recsocial.shared.visualization.gallery import write_figures_index
from recsocial.slices.v3.config import V3Config

V3_HEADLINE = ["B1", "SC", "SCSA", "SC-SCSA_PLUS", "B1-SCSA_PLUS"]


def _paper_targets_for_figures(cfg: V3Config) -> dict[str, dict[str, float]]:
    targets = cfg.paper_targets
    aliases = targets.aliases or {"SCSA_PLUS": "SC"}
    out: dict[str, dict[str, float]] = {}
    for metric in ("mrr", "map", "ndcg"):
        vals = getattr(targets, metric)
        if not vals:
            continue
        remapped = dict(vals)
        for headline, base in aliases.items():
            if base in vals and headline not in remapped:
                remapped[headline] = vals[base]
        out[metric] = remapped
    return out


def generate_v3_figures(reports_dir: Path, cfg: V3Config) -> dict[str, Path]:
    figures_dir = reports_dir / "figures"
    summary = pd.read_csv(reports_dir / "v3_metrics_summary.csv")
    detail = pd.read_csv(reports_dir / "v3_metrics_detail.csv")

    paper_targets = _paper_targets_for_figures(cfg)
    headline = [a for a in V3_HEADLINE if a in set(summary["algorithm"])]

    paths: dict[str, Path] = {
        "headline_metrics": plot_grouped_metrics_bar(
            summary,
            figures_dir / "fig01_headline_metrics.png",
            title="V3 — SCSA-PLUS vs Baselines (Paper §26)",
            algorithms=headline,
            paper_targets=paper_targets,
        ),
        "all_variants": plot_grouped_metrics_bar(
            summary,
            figures_dir / "fig02_all_variants.png",
            title="V3 — All Algorithm Variants",
            algorithms=sorted(summary["algorithm"].tolist()),
        ),
        "precision_at_k": plot_precision_at_k(
            detail,
            figures_dir / "fig03_precision_at_k.png",
            title="V3 — Precision@K (Paper §21.3)",
            algorithms=headline,
        ),
    }

    corr_path = reports_dir / "correlation_matrix.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path, index_col=0)
        paths["correlation"] = plot_correlation_heatmap(
            corr,
            figures_dir / "fig04_correlation_matrix.png",
            title="V3 — Feature Correlation Matrix (Paper §24.1)",
        )

    ttest_path = reports_dir / "paired_ttests.csv"
    if ttest_path.exists():
        ttests = pd.read_csv(ttest_path)
        for i, metric in enumerate(("mrr", "map", "ndcg"), start=5):
            paths[f"ttest_{metric}"] = plot_ttest_heatmap(
                ttests,
                figures_dir / f"fig0{i}_ttest_{metric}.png",
                title="V3 — Paired t-tests (Paper §23)",
                metric=metric,
            )

    shift_path = reports_dir / "ranking_shifts.csv"
    if shift_path.exists():
        shifts = pd.read_csv(shift_path)
        paths["ranking_shifts"] = plot_ranking_shifts(
            shifts,
            figures_dir / "fig08_ranking_shifts.png",
            title="V3 — Ranking Shift vs PCA Score (Paper §24.2)",
            x_col="pca1_score" if "pca1_score" in shifts.columns else "score_orig",
        )

    legacy_path = reports_dir / "legacy_v3_comparison.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        paths["legacy_mrr"] = plot_legacy_comparison(
            legacy,
            figures_dir / "fig09_legacy_mrr.png",
            title="V3 — Reproduction vs Legacy output_v3.csv (MRR)",
            version_prefix="v3",
            metric="mrr",
        )

    gallery = [
        ("fig01_headline_metrics.png", "SCSA-PLUS vs baselines with paper targets"),
        ("fig02_all_variants.png", "Full algorithm variant comparison"),
        ("fig03_precision_at_k.png", "Precision@1–5"),
    ]
    if "correlation" in paths:
        gallery.append(("fig04_correlation_matrix.png", "Feature correlation heatmap"))
    if ttest_path.exists() or "ttest_mrr" in paths:
        gallery.extend(
            [
                ("fig05_ttest_mrr.png", "Paired t-tests — MRR"),
                ("fig06_ttest_map.png", "Paired t-tests — MAP"),
                ("fig07_ttest_ndcg.png", "Paired t-tests — NDCG"),
            ]
        )
    if "ranking_shifts" in paths:
        gallery.append(("fig08_ranking_shifts.png", "PCA ranking shift analysis"))
    if "legacy_mrr" in paths:
        gallery.append(("fig09_legacy_mrr.png", "Legacy validation"))
    write_figures_index(figures_dir, gallery)
    return paths
