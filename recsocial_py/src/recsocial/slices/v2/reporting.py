"""V2 markdown report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.reference_validation import format_validation_report_markdown
from recsocial.shared.reporting import format_metrics_table, write_markdown_report
from recsocial.slices.v2.config import V2Config


def write_v2_report(
    cfg: V2Config,
    summary: pd.DataFrame,
    output_path: Path,
    *,
    validation_result: dict | None = None,
) -> None:
    sections = [
        "# V2 Reproduction Report",
        "",
        "Enhanced Social Capital (AMCIS 2024) — CSV pipeline.",
        "",
        "## Component weights",
        "",
        "```yaml",
    ]
    for k, v in cfg.social_capital.weights.model_dump().items():
        sections.append(f"{k}: {v}")
    sections.extend(["```", "", "## Metrics summary", ""])
    sections.extend(format_metrics_table(summary, sort_by=None))

    if validation_result is not None:
        sections.extend(["", *format_validation_report_markdown(validation_result)])

    sections.extend(
        [
            "",
            "## Figures",
            "",
            f"Publication-style charts: [`figures/index.md`](figures/index.md)",
            "",
            "Regenerate: `python -m recsocial.cli v2 plot`",
            "",
            "## Artifacts",
            "",
            f"- Components: `{cfg.paths.interim_v2_dir}/component_scores.csv`",
            f"- Recommendations: `{cfg.paths.reports_v2_dir}/v2_recommendations.csv`",
            f"- Validation tables: `{cfg.paths.reports_v2_dir}/tables/`",
        ]
    )
    write_markdown_report(output_path, sections)
