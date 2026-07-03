"""Offline experiment runner (SDD §16)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsocial.shared.session_metrics import (
    evaluate_recommendations_by_session,
    settings_from_evaluation_config,
)
from recsocial.slices.v1.config import AppConfig
from recsocial.slices.v1.data_loader import load_processed_bundle
from recsocial.slices.v1.migrate import migrate_to_sdd_schema
from recsocial.shared.algorithms import V1_PAPER_ALGORITHM_ORDER, V1_PAPER_LABELS
from recsocial.slices.v1.recommenders import Recommender
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
from recsocial.slices.v1.text_features import build_hybrid_feature_matrix
from recsocial.slices.v1.user_profile import UserProfileStore


def ensure_processed_data(cfg: AppConfig) -> dict[str, pd.DataFrame]:
    processed_dir = Path(cfg.paths.processed_dir)
    if not (processed_dir / "news.csv").exists():
        migrate_to_sdd_schema(cfg)
    return load_processed_bundle(processed_dir)


def build_artifacts(cfg: AppConfig, data: dict[str, pd.DataFrame]):
    features = build_hybrid_feature_matrix(
        data["news"], data["users"], cfg.text, cfg.pca
    )
    engine = build_score_engine(data["users"], data["news"], data["comments"], cfg)
    sc_scores = score_all_news(engine, data["news"], sentiment_enabled=False)
    scsa_scores = score_all_news(engine, data["news"], sentiment_enabled=True)
    profiles = UserProfileStore(features, cfg.recommendation)
    recommender = Recommender(
        cfg, engine, data["news"], profiles, sc_scores, scsa_scores
    )
    return features, engine, sc_scores, scsa_scores, profiles, recommender


def save_interim_artifacts(
    cfg: AppConfig,
    sc_scores: pd.DataFrame,
    scsa_scores: pd.DataFrame,
    features_matrix: np.ndarray,
    news_ids: list[str],
) -> None:
    interim = Path(cfg.paths.interim_dir)
    interim.mkdir(parents=True, exist_ok=True)
    sc_scores.to_csv(interim / "scored_news_sc.csv", index=False)
    scsa_scores.to_csv(interim / "scored_news_scsa.csv", index=False)
    feat_df = pd.DataFrame(features_matrix, index=news_ids)
    feat_df.index.name = "news_id"
    feat_df.to_csv(interim / "news_feature_matrix.csv")


def evaluate_trial_ratings(cfg: AppConfig, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate stored user-trial rankings (paper reproduction path)."""
    settings = settings_from_evaluation_config(
        cfg.evaluation,
        algorithm_aliases=V1_PAPER_LABELS,
    )
    df = ratings_df.copy().rename(columns={"position": settings.rank_col})
    return evaluate_recommendations_by_session(df, settings)


def run_reranking_experiment(
    cfg: AppConfig,
    data: dict[str, pd.DataFrame],
    recommender: Recommender,
    profiles: UserProfileStore,
) -> pd.DataFrame:
    """Re-score and re-rank trial items; evaluate with stored ratings."""
    ratings = data["ratings"].copy()
    ratings["algorithm"] = ratings["algorithm"].map(
        lambda a: V1_PAPER_LABELS.get(a, a)
    )
    rec_rows: list[dict] = []

    for (uid, algo), session in ratings.groupby(["user_id", "algorithm"]):
        trial_ids = session["news_id"].astype(str).tolist()
        rating_map = session.set_index("news_id")["rating"].astype(int).to_dict()
        profiles.update_from_ratings(uid, trial_ids, list(rating_map.values()))
        recs = recommender.recommend(
            uid, trial_ids, algo, top_k=cfg.recommendation.top_k
        )
        for r in recs:
            rec_rows.append(
                {
                    "user_id": uid,
                    "algorithm": algo,
                    "news_id": r.news_id,
                    "ranking": r.rank,
                    "rating": rating_map[r.news_id],
                    "score": r.score,
                }
            )
    return pd.DataFrame(rec_rows)


def oracle_validation(data: dict[str, pd.DataFrame], sc_scores: pd.DataFrame) -> pd.DataFrame:
    news = data["news"][["news_id", "oracle_social_capital_score"]].copy()
    news["news_id"] = news["news_id"].astype(str)
    scores = sc_scores.copy()
    scores["news_id"] = scores["news_id"].astype(str)
    merged = news.merge(scores, on="news_id", how="inner")
    corr = merged["oracle_social_capital_score"].corr(merged["social_capital_score"])
    rank_corr = merged["oracle_social_capital_score"].rank().corr(
        merged["social_capital_score"].rank()
    )
    merged["abs_error"] = (
        merged["oracle_social_capital_score"] - merged["social_capital_score"]
    ).abs()
    summary = pd.DataFrame(
        [
            {"metric": "pearson_corr", "value": corr},
            {"metric": "spearman_rank_corr", "value": rank_corr},
            {"metric": "mean_abs_error", "value": merged["abs_error"].mean()},
            {"metric": "median_abs_error", "value": merged["abs_error"].median()},
        ]
    )
    return summary


def run_experiment(cfg: AppConfig) -> dict[str, Path]:
    data = ensure_processed_data(cfg)
    features, engine, sc_scores, scsa_scores, profiles, recommender = build_artifacts(
        cfg, data
    )
    save_interim_artifacts(cfg, sc_scores, scsa_scores, features.matrix, features.news_ids)

    trial_metrics = evaluate_trial_ratings(cfg, data["ratings"])
    rerank_df = run_reranking_experiment(cfg, data, recommender, profiles)
    rerank_metrics = evaluate_trial_ratings(cfg, rerank_df) if not rerank_df.empty else trial_metrics

    oracle_summary = oracle_validation(data, sc_scores)

    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    trial_summary = (
        trial_metrics.groupby("algorithm")[["mrr", "map", "ndcg"]].mean().reset_index()
    )
    paths = {
        "trial_metrics_detail": reports_dir / "trial_metrics_detail.csv",
        "trial_metrics_summary": reports_dir / "trial_metrics_summary.csv",
        "rerank_metrics_detail": reports_dir / "rerank_metrics_detail.csv",
        "rerank_metrics_summary": reports_dir / "rerank_metrics_summary.csv",
        "rerank_recommendations": reports_dir / "rerank_recommendations.csv",
        "oracle_validation": reports_dir / "oracle_validation.csv",
        "generated_recommendations": reports_dir / "generated_recommendations.csv",
    }
    trial_metrics.to_csv(paths["trial_metrics_detail"], index=False)
    trial_summary.to_csv(paths["trial_metrics_summary"], index=False)
    rerank_metrics.to_csv(paths["rerank_metrics_detail"], index=False)
    if not rerank_df.empty:
        (
            rerank_metrics.groupby("algorithm")[["mrr", "map", "ndcg"]]
            .mean()
            .reset_index()
            .to_csv(paths["rerank_metrics_summary"], index=False)
        )
    rerank_df.to_csv(paths["rerank_recommendations"], index=False)
    oracle_summary.to_csv(paths["oracle_validation"], index=False)

    gen_rows: list[dict] = []
    all_news = data["news"]["news_id"].astype(str).tolist()
    for uid in data["ratings"]["user_id"].unique():
        for algo in V1_PAPER_ALGORITHM_ORDER:
            for rec in recommender.recommend(uid, all_news, algo):
                gen_rows.append(rec.model_dump())
    pd.DataFrame(gen_rows).to_csv(paths["generated_recommendations"], index=False)

    from recsocial.slices.v1.figures import generate_v1_figures
    from recsocial.slices.v1.reporting import write_report

    write_report(cfg, trial_summary, oracle_summary, reports_dir / "report.md")
    figure_paths = generate_v1_figures(reports_dir, cfg)
    paths.update({f"figure_{k}": v for k, v in figure_paths.items()})
    paths["figures_index"] = reports_dir / "figures" / "index.md"

    return paths
