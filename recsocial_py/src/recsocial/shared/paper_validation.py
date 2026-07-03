"""Unified paper-target validation for V1, V2, and V3 reproduction slices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from recsocial.shared.reference_validation import run_v2_reference_validation
from recsocial.shared.reporting import write_markdown_report
from recsocial.slices.v1.config import load_config
from recsocial.slices.v3.config import load_v3_config


PAPER_TITLES = {
    "v1": "FedCSIS 2022 — Social Capital Recommender",
    "v2": "AMCIS 2024 — Unlocking the Power of Social Capital",
    "v3": "SCSA-PLUS — Enhanced Personalized Recommendations",
}

from recsocial.shared.algorithms import V1_PAPER_LABELS as V1_ALIASES


@dataclass(frozen=True)
class SliceValidationSummary:
    slice_id: str
    paper_title: str
    n_checks: int
    n_pass_strict: int
    n_pass_relaxed: int
    report_path: Path | None
    details_path: Path | None
    status: str  # pass | partial | fail | missing


def _within(actual: float, expected: float, tolerance: float) -> bool:
    return abs(actual - expected) <= tolerance


def _status_from_rates(strict_rate: float, relaxed_rate: float) -> str:
    if strict_rate >= 1.0 or relaxed_rate >= 1.0:
        return "pass"
    if relaxed_rate >= 0.8:
        return "partial"
    if relaxed_rate >= 0.5:
        return "partial"
    return "fail"


def validate_v1_targets(
    package_root: Path,
    *,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, SliceValidationSummary]:
    cfg = load_config(config_path or package_root / "configs" / "v1.yaml", base_dir=package_root)
    summary_path = Path(cfg.paths.reports_dir) / "trial_metrics_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(), SliceValidationSummary(
            slice_id="v1",
            paper_title=PAPER_TITLES["v1"],
            n_checks=0,
            n_pass_strict=0,
            n_pass_relaxed=0,
            report_path=Path(cfg.paths.reports_dir) / "report.md",
            details_path=None,
            status="missing",
        )

    summary = pd.read_csv(summary_path)
    tol = cfg.paper_targets.tolerance
    rows: list[dict[str, Any]] = []

    for algo, row in summary.set_index("algorithm").iterrows():
        canon = V1_ALIASES.get(str(algo), str(algo))
        for metric in ("mrr", "map", "ndcg"):
            targets = getattr(cfg.paper_targets, metric)
            expected = targets.get(canon)
            if expected is None:
                continue
            actual = float(row[metric])
            diff = actual - expected
            rows.append(
                {
                    "slice": "v1",
                    "algorithm": str(algo),
                    "paper_label": canon,
                    "metric": metric.upper(),
                    "expected": expected,
                    "actual": actual,
                    "diff": diff,
                    "pass_strict": _within(actual, expected, tol),
                    "pass_relaxed": _within(actual, expected, tol),
                }
            )

    df = pd.DataFrame(rows)
    strict = int(df["pass_strict"].sum()) if not df.empty else 0
    relaxed = int(df["pass_relaxed"].sum()) if not df.empty else 0
    n = len(df)
    status = _status_from_rates(strict / n if n else 0, relaxed / n if n else 0) if n else "missing"

    return df, SliceValidationSummary(
        slice_id="v1",
        paper_title=PAPER_TITLES["v1"],
        n_checks=n,
        n_pass_strict=strict,
        n_pass_relaxed=relaxed,
        report_path=Path(cfg.paths.reports_dir) / "report.md",
        details_path=summary_path,
        status=status,
    )


def validate_v3_targets(
    package_root: Path,
    *,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, SliceValidationSummary]:
    cfg = load_v3_config(config_path or package_root / "configs" / "v3.yaml", base_dir=package_root)
    summary_path = Path(cfg.paths.reports_v3_dir) / "v3_metrics_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame(), SliceValidationSummary(
            slice_id="v3",
            paper_title=PAPER_TITLES["v3"],
            n_checks=0,
            n_pass_strict=0,
            n_pass_relaxed=0,
            report_path=Path(cfg.paths.reports_v3_dir) / "report.md",
            details_path=None,
            status="missing",
        )

    summary = pd.read_csv(summary_path)
    targets = cfg.paper_targets
    tol = targets.tolerance
    rows: list[dict[str, Any]] = []

    checked: set[tuple[str, str]] = set()
    for headline, base_algo in (targets.aliases or {"SCSA_PLUS": "SC"}).items():
        match = summary[summary["algorithm"] == base_algo]
        if match.empty:
            continue
        row = match.iloc[0]
        for metric in ("mrr", "map", "ndcg"):
            metric_targets = getattr(targets, metric)
            expected = metric_targets.get(base_algo)
            if expected is None:
                continue
            key = (headline, metric)
            if key in checked:
                continue
            checked.add(key)
            actual = float(row[metric])
            diff = actual - expected
            rows.append(
                {
                    "slice": "v3",
                    "algorithm": base_algo,
                    "paper_label": headline,
                    "metric": metric.upper(),
                    "expected": expected,
                    "actual": actual,
                    "diff": diff,
                    "pass_strict": _within(actual, expected, tol),
                    "pass_relaxed": _within(actual, expected, tol),
                }
            )

    for base in ("B1", "SCSA"):
        match = summary[summary["algorithm"] == base]
        if match.empty:
            continue
        row = match.iloc[0]
        for metric in ("mrr", "map", "ndcg"):
            metric_targets = getattr(targets, metric)
            expected = metric_targets.get(base)
            if expected is None:
                continue
            key = (base, metric)
            if key in checked:
                continue
            checked.add(key)
            actual = float(row[metric])
            diff = actual - expected
            rows.append(
                {
                    "slice": "v3",
                    "algorithm": base,
                    "paper_label": base,
                    "metric": metric.upper(),
                    "expected": expected,
                    "actual": actual,
                    "diff": diff,
                    "pass_strict": _within(actual, expected, tol),
                    "pass_relaxed": _within(actual, expected, tol),
                }
            )

    df = pd.DataFrame(rows)
    strict = int(df["pass_strict"].sum()) if not df.empty else 0
    relaxed = int(df["pass_relaxed"].sum()) if not df.empty else 0
    n = len(df)
    status = _status_from_rates(strict / n if n else 0, relaxed / n if n else 0) if n else "missing"

    return df, SliceValidationSummary(
        slice_id="v3",
        paper_title=PAPER_TITLES["v3"],
        n_checks=n,
        n_pass_strict=strict,
        n_pass_relaxed=relaxed,
        report_path=Path(cfg.paths.reports_v3_dir) / "report.md",
        details_path=summary_path,
        status=status,
    )


def validate_v2_targets(package_root: Path) -> tuple[pd.DataFrame, SliceValidationSummary]:
    reference_path = package_root / "configs" / "reference_results.yaml"
    recs_path = package_root / "reports" / "v2" / "v2_recommendations.csv"
    reports_dir = package_root / "reports" / "v2"

    if not recs_path.exists() or not reference_path.exists():
        return pd.DataFrame(), SliceValidationSummary(
            slice_id="v2",
            paper_title=PAPER_TITLES["v2"],
            n_checks=0,
            n_pass_strict=0,
            n_pass_relaxed=0,
            report_path=reports_dir / "report.md",
            details_path=None,
            status="missing",
        )

    recs = pd.read_csv(recs_path)
    result = run_v2_reference_validation(recs, reference_path, reports_dir)
    ranking = result["ranking"]
    precision = result["precision"]
    combined = pd.concat([ranking, precision], ignore_index=True)

    rows: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        rows.append(
            {
                "slice": "v2",
                "figure": row["figure"],
                "algorithm": row["algorithm"],
                "metric": row["metric"],
                "expected": row["expected"],
                "actual": row["actual"],
                "diff": row["diff"],
                "pass_strict": bool(row["pass_strict"]),
                "pass_relaxed": bool(row["pass_relaxed"]),
            }
        )
    df = pd.DataFrame(rows)
    strict = int(df["pass_strict"].sum()) if not df.empty else 0
    relaxed = int(df["pass_relaxed"].sum()) if not df.empty else 0
    n = len(df)
    status = _status_from_rates(strict / n if n else 0, relaxed / n if n else 0) if n else "missing"

    return df, SliceValidationSummary(
        slice_id="v2",
        paper_title=PAPER_TITLES["v2"],
        n_checks=n,
        n_pass_strict=strict,
        n_pass_relaxed=relaxed,
        report_path=reports_dir / "report.md",
        details_path=reports_dir / "tables" / "validation_summary.csv",
        status=status,
    )


def validate_all_papers(package_root: Path | None = None) -> dict[str, Any]:
    root = package_root or Path(__file__).resolve().parents[3]
    v1_df, v1_sum = validate_v1_targets(root)
    v2_df, v2_sum = validate_v2_targets(root)
    v3_df, v3_sum = validate_v3_targets(root)

    summaries = [v1_sum, v2_sum, v3_sum]
    combined = pd.concat([v1_df, v2_df, v3_df], ignore_index=True, sort=False)

    return {
        "v1": {"details": v1_df, "summary": v1_sum},
        "v2": {"details": v2_df, "summary": v2_sum},
        "v3": {"details": v3_df, "summary": v3_sum},
        "summaries": summaries,
        "combined": combined,
    }


def write_cross_paper_validation_report(
    result: dict[str, Any],
    output_path: Path,
) -> Path:
    summaries: list[SliceValidationSummary] = result["summaries"]
    combined: pd.DataFrame = result["combined"]
    root = output_path.parent.parent

    lines = [
        "# Cross-Paper Validation Summary",
        "",
        "Comparison of implemented algorithms against published reference values.",
        "",
        "## Overview",
        "",
        "| Slice | Paper | Checks | Strict pass | Relaxed pass | Status | Report |",
        "|-------|-------|--------|-------------|--------------|--------|--------|",
    ]

    for s in summaries:
        if s.report_path and s.report_path.is_relative_to(root):
            report_link = f"[report]({s.report_path.relative_to(root).as_posix()})"
        elif s.report_path:
            report_link = f"[report]({s.report_path.as_posix()})"
        else:
            report_link = "—"
        lines.append(
            f"| {s.slice_id.upper()} | {s.paper_title} | {s.n_checks} | "
            f"{s.n_pass_strict} | {s.n_pass_relaxed} | **{s.status}** | {report_link} |"
        )

    total_checks = sum(s.n_checks for s in summaries)
    total_strict = sum(s.n_pass_strict for s in summaries)
    total_relaxed = sum(s.n_pass_relaxed for s in summaries)
    lines.extend(
        [
            "",
            f"**Total:** {total_strict}/{total_checks} strict, "
            f"{total_relaxed}/{total_checks} relaxed (± slice-specific tolerances).",
            "",
            "## V1 — FedCSIS trial metrics",
            "",
        ]
    )
    lines.extend(_format_target_table(result["v1"]["details"]))

    lines.extend(["", "## V2 — AMCIS Figures 3–10", ""])
    v2 = result["v2"]["details"]
    if v2.empty:
        lines.append("_No V2 validation data — run `recsocial run v2` first._")
    else:
        lines.append(
            f"V2 uses chart-aligned MAP and binary NDCG. "
            f"Full tables: `reports/v2/tables/`."
        )
        failures = v2[~v2["pass_relaxed"]].head(10)
        if failures.empty:
            lines.append("")
            lines.append("All V2 metric checks within relaxed tolerance.")
        else:
            lines.extend(["", "Largest relaxed-tolerance gaps (sample):", ""])
            lines.extend(_format_v2_gap_table(failures))

    lines.extend(["", "## V3 — SCSA-PLUS §26 headline metrics", ""])
    lines.extend(_format_target_table(result["v3"]["details"]))

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "# Run all slices and regenerate reports",
            "python -m recsocial.cli run all",
            "",
            "# Validate only (requires existing report CSVs)",
            "python -m recsocial.cli validate",
            "```",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(output_path, lines)

    details_csv = output_path.parent / "validation_details.csv"
    if not combined.empty:
        combined.to_csv(details_csv, index=False)

    return output_path


def _format_target_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_No data — run the slice experiment first._"]
    header = "| Algorithm | Metric | Expected | Actual | Δ | Status |"
    sep = "|-----------|--------|----------|--------|---|--------|"
    body = [
        f"| {row['algorithm']} | {row['metric']} | {row['expected']:.3f} | "
        f"{row['actual']:.3f} | {row['diff']:+.3f} | "
        f"{'pass' if row['pass_relaxed'] else 'fail'} |"
        for _, row in df.iterrows()
    ]
    return [header, sep, *body]


def _format_v2_gap_table(df: pd.DataFrame) -> list[str]:
    header = "| Figure | Algorithm | Metric | Expected | Actual | Δ |"
    sep = "|--------|-----------|--------|----------|--------|---|"
    body = [
        f"| {row['figure']} | {row['algorithm']} | {row['metric']} | "
        f"{row['expected']:.3f} | {row['actual']:.3f} | {row['diff']:+.3f} |"
        for _, row in df.iterrows()
    ]
    return [header, sep, *body]
