"""End-to-end pipeline runners for V1/V2/V3 slices."""

from __future__ import annotations

from pathlib import Path

from recsocial.slices.v1.config import load_config
from recsocial.slices.v1.experiment import ensure_processed_data, run_experiment
from recsocial.slices.v1.migrate import migrate_to_sdd_schema
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
from recsocial.slices.v2.config import load_v2_config
from recsocial.slices.v2.experiment import run_v2_experiment
from recsocial.slices.v2.features import enrich_news_for_v2, score_components
from recsocial.slices.v3.config import load_v3_config
from recsocial.slices.v3.experiment import run_v3_experiment
from recsocial.slices.v3.features import build_v3_features


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_v1_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    migrate_to_sdd_schema(cfg)
    data = ensure_processed_data(cfg)
    engine = build_score_engine(data["users"], data["news"], data["comments"], cfg)
    out = Path(cfg.paths.interim_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_all_news(engine, data["news"], sentiment_enabled=False).to_csv(out / "scored_news_sc.csv", index=False)
    score_all_news(engine, data["news"], sentiment_enabled=True).to_csv(out / "scored_news_scsa.csv", index=False)
    return run_experiment(cfg)


def run_v2_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    v1_cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    cfg = load_v2_config(root / "configs" / "v2.yaml", base_dir=root)
    migrate_to_sdd_schema(v1_cfg)
    enrich_news_for_v2(cfg, v1_cfg)
    score_components(cfg, v1_cfg)
    return run_v2_experiment(cfg, root)


def run_v3_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    v1_cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    v2_cfg = load_v2_config(root / "configs" / "v2.yaml", base_dir=root)
    cfg = load_v3_config(root / "configs" / "v3.yaml", base_dir=root)
    migrate_to_sdd_schema(v1_cfg)
    enrich_news_for_v2(v2_cfg, v1_cfg)
    build_v3_features(cfg, root)
    return run_v3_experiment(cfg, root)


def run_all_pipelines(root: Path | None = None) -> dict[str, dict[str, Path]]:
    """Run V1 → V2 → V3 with shared preprocessing done once."""
    root = root or package_root()
    v1_cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    v2_cfg = load_v2_config(root / "configs" / "v2.yaml", base_dir=root)
    v3_cfg = load_v3_config(root / "configs" / "v3.yaml", base_dir=root)

    migrate_to_sdd_schema(v1_cfg)
    enrich_news_for_v2(v2_cfg, v1_cfg)
    build_v3_features(v3_cfg, root)

    data = ensure_processed_data(v1_cfg)
    engine = build_score_engine(data["users"], data["news"], data["comments"], v1_cfg)
    interim = Path(v1_cfg.paths.interim_dir)
    interim.mkdir(parents=True, exist_ok=True)
    score_all_news(engine, data["news"], sentiment_enabled=False).to_csv(
        interim / "scored_news_sc.csv", index=False
    )
    score_all_news(engine, data["news"], sentiment_enabled=True).to_csv(
        interim / "scored_news_scsa.csv", index=False
    )
    v1_paths = run_experiment(v1_cfg)

    score_components(v2_cfg, v1_cfg)
    v2_paths = run_v2_experiment(v2_cfg, root)

    v3_paths = run_v3_experiment(v3_cfg, root)
    return {"v1": v1_paths, "v2": v2_paths, "v3": v3_paths}
