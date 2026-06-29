"""CLI — vertical slice entrypoints for V1, V2, and V3."""

from __future__ import annotations

import argparse
from pathlib import Path

from recsocial.shared.paper_validation import validate_all_papers, write_cross_paper_validation_report
from recsocial.shared.pipeline import (
    package_root,
    run_all_pipelines,
    run_v1_pipeline,
    run_v2_pipeline,
    run_v3_pipeline,
)
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
from recsocial.slices.v1.figures import generate_v1_figures
from recsocial.slices.v2.figures import generate_v2_figures
from recsocial.slices.v3.figures import generate_v3_figures


def _package_root() -> Path:
    return package_root()


def cmd_run(args: argparse.Namespace) -> None:
    root = _package_root()
    slice_name = args.slice
    if slice_name == "all":
        paths = run_all_pipelines(root)
        for sl, sl_paths in paths.items():
            print(f"\n=== {sl.upper()} ===")
            for name, path in sl_paths.items():
                print(f"{name}: {path}")
    elif slice_name == "v1":
        for name, path in run_v1_pipeline(root).items():
            print(f"{name}: {path}")
    elif slice_name == "v2":
        for name, path in run_v2_pipeline(root).items():
            print(f"{name}: {path}")
    elif slice_name == "v3":
        for name, path in run_v3_pipeline(root).items():
            print(f"{name}: {path}")

    if args.validate or slice_name == "all":
        cmd_validate(args)


def cmd_validate(args: argparse.Namespace) -> None:
    root = _package_root()
    result = validate_all_papers(root)
    out = root / "reports" / "validation_summary.md"
    write_cross_paper_validation_report(result, out)
    print(f"\nValidation summary: {out}")

    for s in result["summaries"]:
        rate = f"{s.n_pass_relaxed}/{s.n_checks}" if s.n_checks else "n/a"
        print(f"  {s.slice_id.upper()}: {s.status} ({rate} relaxed)")


def _add_v1_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(_package_root() / "configs" / "v1.yaml"))


def _add_v2_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(_package_root() / "configs" / "v2.yaml"))


def _add_v3_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(_package_root() / "configs" / "v3.yaml"))


# --- V1 ---

def cmd_v1_preprocess(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, base_dir=_package_root())
    for name, path in migrate_to_sdd_schema(cfg).items():
        print(f"{name}: {path}")


def cmd_v1_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, base_dir=_package_root())
    data = ensure_processed_data(cfg)
    engine = build_score_engine(data["users"], data["news"], data["comments"], cfg)
    out = Path(cfg.paths.interim_dir)
    out.mkdir(parents=True, exist_ok=True)
    score_all_news(engine, data["news"], sentiment_enabled=False).to_csv(out / "scored_news_sc.csv", index=False)
    score_all_news(engine, data["news"], sentiment_enabled=True).to_csv(out / "scored_news_scsa.csv", index=False)
    print(f"Wrote scores to {out}")


def cmd_v1_experiment(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, base_dir=_package_root())
    for name, path in run_experiment(cfg).items():
        print(f"{name}: {path}")


def cmd_v1_plot(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, base_dir=_package_root())
    paths = generate_v1_figures(Path(cfg.paths.reports_dir), cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"index: {Path(cfg.paths.reports_dir) / 'figures' / 'index.md'}")


def cmd_v1_report(args: argparse.Namespace) -> None:
    import pandas as pd

    from recsocial.slices.v1.reporting import write_report

    cfg = load_config(args.config, base_dir=_package_root())
    reports = Path(cfg.paths.reports_dir)
    summary = reports / "trial_metrics_summary.csv"
    oracle = reports / "oracle_validation.csv"
    if not summary.exists():
        raise SystemExit("Run v1 experiment first.")
    write_report(cfg, pd.read_csv(summary), pd.read_csv(oracle), reports / "report.md")
    print(f"Report: {reports / 'report.md'}")


# --- V2 ---

def cmd_v2_preprocess(args: argparse.Namespace) -> None:
    cfg = load_v2_config(args.config, base_dir=_package_root())
    path = enrich_news_for_v2(cfg, cfg.load_v1(_package_root()))
    print(f"news_enriched: {Path(cfg.paths.processed_v2_dir) / 'news_enriched.csv'} ({len(path)} rows)")


def cmd_v2_score(args: argparse.Namespace) -> None:
    cfg = load_v2_config(args.config, base_dir=_package_root())
    frame = score_components(cfg, cfg.load_v1(_package_root()))
    print(f"Wrote {len(frame.df)} rows to {cfg.paths.interim_v2_dir}/component_scores.csv")


def cmd_v2_plot(args: argparse.Namespace) -> None:
    import pandas as pd

    from recsocial.shared.statistics import run_paired_t_tests

    cfg = load_v2_config(args.config, base_dir=_package_root())
    reports = Path(cfg.paths.reports_v2_dir)
    ttest_path = reports / "paired_ttests.csv"
    if ttest_path.exists():
        ttests = pd.read_csv(ttest_path)
    elif cfg.statistics.paired_t_test and (reports / "v2_metrics_detail.csv").exists():
        detail = pd.read_csv(reports / "v2_metrics_detail.csv")
        ttests = run_paired_t_tests(detail, cfg.statistics)
        ttests.to_csv(ttest_path, index=False)
    else:
        ttests = None
    paths = generate_v2_figures(reports, cfg, ttests=ttests)
    for name, path in paths.items():
        print(f"{name}: {path}")


def cmd_v2_experiment(args: argparse.Namespace) -> None:
    cfg = load_v2_config(args.config, base_dir=_package_root())
    for name, path in run_v2_experiment(cfg, _package_root()).items():
        print(f"{name}: {path}")


# --- V3 ---

def cmd_v3_preprocess(args: argparse.Namespace) -> None:
    cfg = load_v3_config(args.config, base_dir=_package_root())
    v2_cfg = load_v2_config(cfg.paths.v2_config_path, base_dir=_package_root())
    enrich_news_for_v2(v2_cfg, cfg.load_v1(_package_root()))
    print(f"Prepared V2 inputs for V3 under {cfg.paths.processed_v3_dir}")


def cmd_v3_score(args: argparse.Namespace) -> None:
    cfg = load_v3_config(args.config, base_dir=_package_root())
    df = build_v3_features(cfg, _package_root())
    print(f"Wrote {len(df)} feature rows to {cfg.paths.interim_v3_dir}/v3_feature_scores.csv")


def cmd_v3_plot(args: argparse.Namespace) -> None:
    cfg = load_v3_config(args.config, base_dir=_package_root())
    paths = generate_v3_figures(Path(cfg.paths.reports_v3_dir), cfg)
    for name, path in paths.items():
        print(f"{name}: {path}")


def cmd_v3_experiment(args: argparse.Namespace) -> None:
    cfg = load_v3_config(args.config, base_dir=_package_root())
    for name, path in run_v3_experiment(cfg, _package_root()).items():
        print(f"{name}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="recsocial", description="Social Capital recommender (V1/V2/V3)")
    sub = parser.add_subparsers(dest="slice", required=True)

    v1 = sub.add_parser("v1", help="V1 paper baseline (FedCSIS 2022)")
    v1_sub = v1.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("preprocess", cmd_v1_preprocess, "Migrate raw CSVs to SDD schema"),
        ("score", cmd_v1_score, "Compute SC and SC+SA scores"),
        ("experiment", cmd_v1_experiment, "Run V1 experiment"),
        ("report", cmd_v1_report, "Regenerate markdown report"),
        ("plot", cmd_v1_plot, "Generate paper-style figures from report CSVs"),
    ):
        p = v1_sub.add_parser(name, help=help_text)
        _add_v1_config(p)
        p.set_defaults(func=func)

    v2 = sub.add_parser("v2", help="V2 enhanced Social Capital (AMCIS 2024)")
    v2_sub = v2.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("preprocess", cmd_v2_preprocess, "Enrich news for V2"),
        ("score", cmd_v2_score, "Compute V2 component scores"),
        ("experiment", cmd_v2_experiment, "Run V2 experiment"),
        ("plot", cmd_v2_plot, "Generate paper-style figures from report CSVs"),
    ):
        p = v2_sub.add_parser(name, help=help_text)
        _add_v2_config(p)
        p.set_defaults(func=func)

    v3 = sub.add_parser("v3", help="V3 SCSA-PLUS + PCA")
    v3_sub = v3.add_subparsers(dest="command", required=True)
    for name, func, help_text in (
        ("preprocess", cmd_v3_preprocess, "Prepare enriched CSVs for V3"),
        ("score", cmd_v3_score, "Compute SCSA-PLUS and PCA scores"),
        ("experiment", cmd_v3_experiment, "Run V3 experiment"),
        ("plot", cmd_v3_plot, "Generate paper-style figures from report CSVs"),
    ):
        p = v3_sub.add_parser(name, help=help_text)
        _add_v3_config(p)
        p.set_defaults(func=func)

    run_p = sub.add_parser("run", help="Run full pipeline (preprocess → score → experiment → report)")
    run_p.add_argument(
        "slice",
        choices=["all", "v1", "v2", "v3"],
        help="Which slice to run (all runs V1→V2→V3 with shared prep)",
    )
    run_p.add_argument(
        "--validate",
        action="store_true",
        help="Also run cross-paper validation after the pipeline",
    )
    run_p.set_defaults(func=cmd_run)

    val_p = sub.add_parser("validate", help="Validate all slices against paper reference values")
    val_p.add_argument("--slice", choices=["all"], default="all", help=argparse.SUPPRESS)
    val_p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
