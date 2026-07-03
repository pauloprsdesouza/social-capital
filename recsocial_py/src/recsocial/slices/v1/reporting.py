"""Markdown reproduction report (SDD §19.4)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.reporting import compare_target_row, format_metrics_table, write_markdown_report
from recsocial.slices.v1.config import AppConfig


def write_report(
    cfg: AppConfig,
    trial_summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    tolerance = cfg.paper_targets.tolerance
    sections = [
        "# V1 Reproduction Report",
        "",
        "CSV-only pipeline — no database.",
        "",
        "## Trial metrics (stored user ratings)",
        "",
        *format_metrics_table(trial_summary, sort_by=None),
        "",
        "## Comparison vs paper targets",
        "",
        "| Algorithm | Metric | Measured | Paper | Delta | Status |",
        "|-----------|--------|----------|-------|-------|--------|",
    ]
    for row in trial_summary.itertuples(index=False):
        for metric in ("mrr", "map", "ndcg"):
            targets = getattr(cfg.paper_targets, metric)
            sections.append(
                compare_target_row(
                    row.algorithm, metric.upper(), getattr(row, metric), targets, tolerance
                )
            )

    sections.extend(["", "## Oracle SC score validation", ""])
    for row in oracle_summary.itertuples(index=False):
        val = row.value
        if pd.notna(val) and isinstance(val, float):
            sections.append(f"- **{row.metric}**: {val:.4f}")
        else:
            sections.append(f"- **{row.metric}**: {val}")

    sections.extend(
        [
            "",
            "## Figures",
            "",
            f"Publication-style charts: [`figures/index.md`](figures/index.md)",
            "",
            "Regenerate without re-running the experiment:",
            "",
            "```bash",
            "python -m recsocial.cli v1 plot",
            "```",
            "",
            "## Assumptions",
            "",
            "See `docs/VERSIONS.md#v1-implementation-notes` for V1 defaults.",
            "",
            "## Data artifacts",
            "",
            f"- Processed CSVs: `{cfg.paths.processed_dir}`",
            f"- Interim scores: `{cfg.paths.interim_dir}`",
            f"- Reports: `{cfg.paths.reports_dir}`",
        ]
    )
    write_markdown_report(output_path, sections)
