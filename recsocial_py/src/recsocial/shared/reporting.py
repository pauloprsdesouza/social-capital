"""Markdown report helpers shared across slices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_metrics_table(summary: pd.DataFrame, *, sort_by: str | None = "algorithm") -> list[str]:
    lines = [
        "| Algorithm | MRR | MAP | NDCG |",
        "|-----------|-----|-----|------|",
    ]
    frame = summary.sort_values(sort_by) if sort_by and sort_by in summary.columns else summary
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.algorithm} | {row.mrr:.3f} | {row.map:.3f} | {row.ndcg:.3f} |")
    return lines


def compare_target_row(
    algo: str,
    metric: str,
    measured: float,
    targets: dict[str, float],
    tolerance: float,
) -> str:
    target = targets.get(algo)
    if target is None:
        return f"| {algo} | {metric} | {measured:.3f} | — | — | n/a |"
    delta = measured - target
    status = "pass" if abs(delta) <= tolerance else "review"
    return f"| {algo} | {metric} | {measured:.3f} | {target:.2f} | {delta:+.3f} | {status} |"


def write_markdown_report(output_path: Path, sections: list[str]) -> None:
    output_path.write_text("\n".join(sections), encoding="utf-8")
