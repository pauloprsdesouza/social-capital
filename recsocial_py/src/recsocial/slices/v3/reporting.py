"""V3 reproduction report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.reporting import format_metrics_table, write_markdown_report
from recsocial.slices.v3.config import V3Config


def write_v3_report(
    cfg: V3Config,
    summary: pd.DataFrame,
    ttests: pd.DataFrame,
    output_path: Path,
) -> None:
    sections = [
        "# V3 Reproduction Report — SCSA-PLUS",
        "",
        "Paper: *Exploiting Social Capital for Improving Personalized Recommendations in Online Social Networks*",
        "",
        "Pipeline: SCSA-PLUS social capital + PCA re-ranking. CSV-only.",
        "",
        "## Metrics summary",
        "",
        *format_metrics_table(summary),
    ]

    if not ttests.empty:
        sections.extend(["", "## Paired t-tests (p ≤ 0.05 significant)", ""])
        sig = ttests[ttests["significant"] == True]  # noqa: E712
        for row in sig.itertuples(index=False):
            sections.append(
                f"- **{row.metric}** {row.algorithm_a} vs {row.algorithm_b}: "
                f"p={row.p_value:.4g} (n={row.n_pairs})"
            )

    targets = cfg.paper_targets
    if targets.mrr or targets.map or targets.ndcg:
        sections.extend(["", "## Paper targets (§26 — base trial algorithms)", ""])
        for metric in ("mrr", "map", "ndcg"):
            vals = getattr(targets, metric)
            if vals:
                sections.append(
                    f"**{metric.upper()}**: " + ", ".join(f"{k}={v}" for k, v in vals.items())
                )
        if targets.aliases:
            sections.append("")
            sections.append(
                "Note: paper headline **SCSA-PLUS** maps to base algorithm "
                f"**{targets.aliases.get('SCSA_PLUS', 'SC')}** on stored trial rankings."
            )

    base_algo = targets.aliases.get("SCSA_PLUS", "SC")
    headline = summary[summary["algorithm"] == base_algo]
    if not headline.empty and targets.mrr:
        sections.extend(["", f"## Paper headline vs measured ({base_algo})", ""])
        row = headline.iloc[0]
        for metric in ("mrr", "map", "ndcg"):
            target_vals = getattr(targets, metric)
            target = target_vals.get(base_algo)
            if target is not None:
                measured = float(row[metric])
                delta = measured - target
                sections.append(
                    f"- **{metric.upper()}** measured **{measured:.3f}** vs paper **{target:.3f}** (Δ {delta:+.3f})"
                )

    sections.extend(
        [
            "",
            "## Figures",
            "",
            f"Publication-style charts: [`figures/index.md`](figures/index.md)",
            "",
            "Regenerate: `python -m recsocial.cli v3 plot`",
            "",
            "## Assumptions",
            "",
            "See `docs/v3/reproduction_notes_v3.md`.",
            "",
            "## Artifacts",
            "",
            f"- Features: `{cfg.paths.interim_v3_dir}/v3_feature_scores.csv`",
            f"- Recommendations: `{cfg.paths.reports_v3_dir}/v3_recommendations.csv`",
        ]
    )
    write_markdown_report(output_path, sections)
