"""Shared experiment orchestration: evaluate, persist artifacts, report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass
class ExperimentArtifacts:
    recommendations: pd.DataFrame
    metrics_detail: pd.DataFrame
    metrics_summary: pd.DataFrame
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)


def persist_experiment_outputs(
    paths: dict[str, Path],
    artifacts: ExperimentArtifacts,
) -> dict[str, Path]:
    reports_dir = next(iter(paths.values())).parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    artifacts.recommendations.to_csv(paths["recommendations"], index=False)
    artifacts.metrics_detail.to_csv(paths["metrics_detail"], index=False)
    artifacts.metrics_summary.to_csv(paths["metrics_summary"], index=False)

    for key, frame in artifacts.extras.items():
        out = paths.get(key)
        if out is None or frame.empty:
            continue
        if out.suffix == ".csv":
            frame.to_csv(out, index=False)
        else:
            frame.to_csv(out)

    return paths


def run_standard_experiment(
    *,
    build_recommendations: Callable[[], pd.DataFrame],
    evaluate: Callable[[pd.DataFrame], pd.DataFrame],
    summarize: Callable[[pd.DataFrame], pd.DataFrame],
    output_paths: dict[str, Path],
    enrich: Callable[[ExperimentArtifacts], ExperimentArtifacts] | None = None,
    write_report: Callable[[ExperimentArtifacts, Path], None] | None = None,
) -> dict[str, Path]:
    recommendations = build_recommendations()
    metrics_detail = evaluate(recommendations)
    metrics_summary = summarize(metrics_detail)
    artifacts = ExperimentArtifacts(recommendations, metrics_detail, metrics_summary)
    if enrich:
        artifacts = enrich(artifacts)
    persist_experiment_outputs(output_paths, artifacts)
    if write_report and "report" in output_paths:
        write_report(artifacts, output_paths["report"])
    return output_paths
