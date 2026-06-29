"""Smoke tests for publication-style figure generation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

from recsocial.shared.visualization.charts import (
    plot_grouped_metrics_bar,
    plot_oracle_validation,
    plot_precision_at_k,
    plot_ttest_heatmap,
)
from recsocial.slices.v1.config import load_config
from recsocial.slices.v1.figures import generate_v1_figures
from recsocial.slices.v2.config import load_v2_config
from recsocial.slices.v2.figures import generate_v2_figures
from recsocial.slices.v3.config import load_v3_config
from recsocial.slices.v3.figures import generate_v3_figures


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def test_chart_helpers_write_png(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "algorithm": ["B1", "SC"],
            "mrr": [0.7, 0.8],
            "map": [0.65, 0.75],
            "ndcg": [0.72, 0.78],
        }
    )
    detail = pd.DataFrame(
        {
            "algorithm": ["B1", "B1", "SC", "SC"],
            "precision_at_1": [0.5, 0.6, 0.7, 0.8],
            "precision_at_2": [0.4, 0.5, 0.6, 0.7],
        }
    )
    oracle = pd.DataFrame({"metric": ["pearson_corr"], "value": [0.95]})
    ttests = pd.DataFrame(
        {
            "metric": ["mrr"],
            "algorithm_a": ["A"],
            "algorithm_b": ["B"],
            "p_value": [0.01],
            "significant": [True],
        }
    )

    for fn, args in (
        (plot_grouped_metrics_bar, (summary, tmp_path / "bar.png")),
        (plot_precision_at_k, (detail, tmp_path / "prec.png")),
        (plot_oracle_validation, (oracle, tmp_path / "oracle.png")),
        (plot_ttest_heatmap, (ttests, tmp_path / "ttest.png")),
    ):
        path = fn(*args, title="Test") if fn != plot_ttest_heatmap else fn(*args, title="Test", metric="mrr")
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.parametrize(
    ("loader", "reports_subdir", "generator"),
    [
        (lambda: load_config(PACKAGE_ROOT / "configs" / "v1.yaml", base_dir=PACKAGE_ROOT), "v1", generate_v1_figures),
        (lambda: load_v2_config(PACKAGE_ROOT / "configs" / "v2.yaml", base_dir=PACKAGE_ROOT), "v2", generate_v2_figures),
        (lambda: load_v3_config(PACKAGE_ROOT / "configs" / "v3.yaml", base_dir=PACKAGE_ROOT), "v3", generate_v3_figures),
    ],
)
def test_slice_figure_gallery(loader, reports_subdir: str, generator) -> None:
    cfg = loader()
    reports_dir = PACKAGE_ROOT / "reports" / reports_subdir
    if not (reports_dir / f"{reports_subdir}_metrics_summary.csv" if reports_subdir != "v1" else reports_dir / "trial_metrics_summary.csv").exists():
        pytest.skip(f"Missing report CSVs under {reports_dir}")

    if reports_subdir == "v1":
        paths = generator(reports_dir, cfg)
    elif reports_subdir == "v2":
        ttest_path = reports_dir / "paired_ttests.csv"
        ttests = pd.read_csv(ttest_path) if ttest_path.exists() else None
        paths = generator(reports_dir, cfg, ttests=ttests)
    else:
        paths = generator(reports_dir, cfg)

    assert paths
    for path in paths.values():
        assert path.exists()
        assert path.suffix == ".png"
    index = reports_dir / "figures" / "index.md"
    assert index.exists()
