"""End-to-end pipeline runners for V1/V2/V3 slices."""

from __future__ import annotations

from pathlib import Path

from recsocial.slices.v1.config import AppConfig, load_config
from recsocial.slices.v1.experiment import ensure_processed_data, run_experiment
from recsocial.slices.v1.migrate import migrate_to_sdd_schema
from recsocial.slices.v1.social_capital import build_score_engine, score_all_news
from recsocial.slices.v2.config import V2Config, load_v2_config
from recsocial.slices.v2.experiment import run_v2_experiment
from recsocial.slices.v2.features import enrich_news_for_v2, score_components
from recsocial.slices.v3.config import V3Config, load_v3_config
from recsocial.slices.v3.experiment import run_v3_experiment


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def score_v1_interim(cfg: AppConfig) -> Path:
    """Compute SC and SC+SA interim score CSVs (V1 score stage)."""
    data = ensure_processed_data(cfg)
    engine = build_score_engine(data["users"], data["news"], data["comments"], cfg)
    out = Path(cfg.paths.interim_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_all_news(engine, data["news"], sentiment_enabled=False).to_csv(
        out / "scored_news_sc.csv", index=False
    )
    score_all_news(engine, data["news"], sentiment_enabled=True).to_csv(
        out / "scored_news_scsa.csv", index=False
    )
    return out


def preprocess_v2(v2_cfg: V2Config, root: Path) -> Path:
    """Migrate V1 raw data and enrich news for V2."""
    v1_cfg = v2_cfg.load_v1(root)
    migrate_to_sdd_schema(v1_cfg)
    return enrich_news_for_v2(v2_cfg, v1_cfg)


def score_v2_components(v2_cfg: V2Config, root: Path):
    """Compute V2 component scores (V2 score stage)."""
    return score_components(v2_cfg, v2_cfg.load_v1(root))


def prepare_v3_inputs(v3_cfg: V3Config, root: Path) -> None:
    """Ensure V1/V2 inputs exist for V3 feature scoring."""
    v1_cfg = v3_cfg.load_v1(root)
    v2_cfg = load_v2_config(v3_cfg.paths.v2_config_path, base_dir=root)
    migrate_to_sdd_schema(v1_cfg)
    enrich_news_for_v2(v2_cfg, v1_cfg)


def run_v1_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    migrate_to_sdd_schema(cfg)
    return run_experiment(cfg)


def run_v2_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    cfg = load_v2_config(root / "configs" / "v2.yaml", base_dir=root)
    preprocess_v2(cfg, root)
    return run_v2_experiment(cfg, root)


def run_v3_pipeline(root: Path | None = None) -> dict[str, Path]:
    root = root or package_root()
    cfg = load_v3_config(root / "configs" / "v3.yaml", base_dir=root)
    prepare_v3_inputs(cfg, root)
    return run_v3_experiment(cfg, root)


def run_all_pipelines(root: Path | None = None) -> dict[str, dict[str, Path]]:
    """Run V1 → V2 → V3 with shared preprocessing done once per dependency chain."""
    root = root or package_root()
    v1_cfg = load_config(root / "configs" / "v1.yaml", base_dir=root)
    v2_cfg = load_v2_config(root / "configs" / "v2.yaml", base_dir=root)
    v3_cfg = load_v3_config(root / "configs" / "v3.yaml", base_dir=root)

    migrate_to_sdd_schema(v1_cfg)
    enrich_news_for_v2(v2_cfg, v1_cfg)

    return {
        "v1": run_experiment(v1_cfg),
        "v2": run_v2_experiment(v2_cfg, root),
        "v3": run_v3_experiment(v3_cfg, root),
    }
