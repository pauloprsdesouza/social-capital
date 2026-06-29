"""V2 experiment runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsocial.shared.experiment_runner import ExperimentArtifacts, run_standard_experiment
from recsocial.shared.legacy_compare import compare_summary_to_legacy
from recsocial.shared.reference_validation import run_v2_reference_validation
from recsocial.shared.session_metrics import (
    evaluate_recommendations_by_session,
    settings_from_evaluation_config,
    summarize_session_metrics,
)
from recsocial.shared.statistics import run_paired_t_tests
from recsocial.slices.v1.experiment import ensure_processed_data
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
from recsocial.slices.v2.config import V2Config
from recsocial.slices.v2.features import score_components
from recsocial.slices.v2.figures import generate_v2_figures
from recsocial.slices.v2.recommenders import build_reranked_recommendations
from recsocial.slices.v2.reporting import write_v2_report


def run_v2_experiment(cfg: V2Config, package_root: Path) -> dict[str, Path]:
    v1_cfg = cfg.load_v1(package_root)
    ensure_processed_data(v1_cfg)

    frame = score_components(cfg, v1_cfg)
    components = frame.df
    ratings = pd.read_csv(Path(cfg.paths.processed_v2_dir) / "ratings.csv")
    ratings["news_id"] = ratings["news_id"].astype(str)
    components["news_id"] = components["news_id"].astype(str)

    news_v1 = pd.read_csv(Path(v1_cfg.paths.processed_dir) / "news.csv")
    users_v2 = pd.read_csv(Path(cfg.paths.processed_v2_dir) / "users.csv")
    comments_v1 = pd.read_csv(Path(v1_cfg.paths.processed_dir) / "comments.csv")
    engine = build_score_engine(users_v2, news_v1, comments_v1, v1_cfg)
    v1_scsa = score_all_news(engine, news_v1, sentiment_enabled=True)
    v1_scsa["news_id"] = v1_scsa["news_id"].astype(str)

    eval_settings = settings_from_evaluation_config(cfg.evaluation)
    reports_dir = Path(cfg.paths.reports_v2_dir)
    output_paths = {
        "recommendations": reports_dir / "v2_recommendations.csv",
        "metrics_detail": reports_dir / "v2_metrics_detail.csv",
        "metrics_summary": reports_dir / "v2_metrics_summary.csv",
        "legacy_comparison": reports_dir / "legacy_metrics_comparison.csv",
        "ttests": reports_dir / "paired_ttests.csv",
        "report": reports_dir / "report.md",
        "validation_tables": reports_dir / "tables",
    }
    legacy_path = Path(cfg.paths.legacy_recommendations)
    reference_path = package_root / "configs" / "reference_results.yaml"

    def _enrich(artifacts: ExperimentArtifacts) -> ExperimentArtifacts:
        extras = dict(artifacts.extras)
        if cfg.statistics.paired_t_test:
            extras["ttests"] = run_paired_t_tests(artifacts.metrics_detail, cfg.statistics)
        if legacy_path.exists():
            extras["legacy_comparison"] = compare_summary_to_legacy(
                artifacts.metrics_summary,
                legacy_path,
                version_prefix="v2",
            )
        if reference_path.exists():
            extras["validation"] = run_v2_reference_validation(
                artifacts.recommendations,
                reference_path,
                reports_dir,
                relevance_threshold=cfg.evaluation.relevance_threshold,
                ndcg_k=cfg.evaluation.ndcg_k,
            )
        artifacts.extras = extras
        return artifacts

    paths = run_standard_experiment(
        build_recommendations=lambda: build_reranked_recommendations(
            ratings, components, cfg, v1_scsa
        ),
        evaluate=lambda recs: evaluate_recommendations_by_session(recs, eval_settings),
        summarize=summarize_session_metrics,
        output_paths=output_paths,
        enrich=_enrich,
        write_report=lambda art, path: write_v2_report(
            cfg,
            art.metrics_summary,
            art.extras.get("legacy_comparison"),
            path,
            validation_result=art.extras.get("validation"),
        ),
    )

    ttests = pd.read_csv(paths["ttests"]) if paths["ttests"].exists() else pd.DataFrame()
    figure_paths = generate_v2_figures(reports_dir, cfg, ttests=ttests if not ttests.empty else None)
    paths.update({f"figure_{k}": v for k, v in figure_paths.items()})
    paths["figures_index"] = reports_dir / "figures" / "index.md"
    return paths
