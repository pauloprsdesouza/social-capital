"""V3 experiment runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.experiment_runner import ExperimentArtifacts, run_standard_experiment
from recsocial.shared.paper_rankings import load_recommendations_or_compute
from recsocial.shared.session_metrics import (
    evaluate_recommendations_by_session,
    settings_from_evaluation_config,
    summarize_session_metrics,
)
from recsocial.slices.v3.config import V3Config
from recsocial.slices.v3.features import build_v3_features
from recsocial.slices.v3.recommenders import build_v3_recommendations
from recsocial.slices.v3.figures import generate_v3_figures
from recsocial.slices.v3.reporting import write_v3_report
from recsocial.shared.statistics import run_paired_t_tests
from recsocial.slices.v3.statistics import correlation_matrix, ranking_shift_analysis


def run_v3_experiment(cfg: V3Config, package_root: Path) -> dict[str, Path]:
    features = build_v3_features(cfg, package_root)
    ratings = pd.read_csv(Path(cfg.paths.processed_v3_dir) / "ratings.csv")
    ratings["news_id"] = ratings["news_id"].astype(str)

    eval_settings = settings_from_evaluation_config(cfg.evaluation)
    reports_dir = Path(cfg.paths.reports_v3_dir)
    output_paths = {
        "recommendations": reports_dir / "v3_recommendations.csv",
        "metrics_detail": reports_dir / "v3_metrics_detail.csv",
        "metrics_summary": reports_dir / "v3_metrics_summary.csv",
        "ttests": reports_dir / "paired_ttests.csv",
        "correlations": reports_dir / "correlation_matrix.csv",
        "ranking_shifts": reports_dir / "ranking_shifts.csv",
        "report": reports_dir / "report.md",
    }

    def _enrich(artifacts: ExperimentArtifacts) -> ExperimentArtifacts:
        extras = dict(artifacts.extras)
        if cfg.statistics.paired_t_test:
            extras["ttests"] = run_paired_t_tests(artifacts.metrics_detail, cfg.statistics)
        extras["correlations"] = correlation_matrix(features)

        shift_frames = []
        for base in cfg.base_algorithms:
            orig_algo = f"{base}-{cfg.rerank_suffixes['state_art']}"
            new_algo = f"{orig_algo}-{cfg.rerank_suffixes['scsa_plus_v3']}"
            orig = artifacts.recommendations[artifacts.recommendations["algorithm"] == orig_algo]
            new = artifacts.recommendations[artifacts.recommendations["algorithm"] == new_algo]
            if not orig.empty and not new.empty:
                shift = ranking_shift_analysis(orig, new)
                shift["base_algorithm"] = base
                shift_frames.append(shift)
        if shift_frames:
            extras["ranking_shifts"] = pd.concat(shift_frames, ignore_index=True)

        artifacts.extras = extras
        return artifacts

    paths = run_standard_experiment(
        build_recommendations=lambda: load_recommendations_or_compute(
            cfg,
            package_root,
            lambda: build_v3_recommendations(ratings, features, cfg),
        ),
        evaluate=lambda recs: evaluate_recommendations_by_session(recs, eval_settings),
        summarize=summarize_session_metrics,
        output_paths=output_paths,
        enrich=_enrich,
        write_report=lambda art, path: write_v3_report(
            cfg,
            art.metrics_summary,
            art.extras.get("ttests", pd.DataFrame()),
            path,
        ),
    )

    figure_paths = generate_v3_figures(reports_dir, cfg)
    paths.update({f"figure_{k}": v for k, v in figure_paths.items()})
    paths["figures_index"] = reports_dir / "figures" / "index.md"
    return paths
