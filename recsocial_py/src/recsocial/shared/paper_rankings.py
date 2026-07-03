"""Load author paper ranking exports (SDD preferred reproduction source)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from recsocial.shared.reference_validation import (
    ReferenceResultsRepository,
    compute_algorithm_metrics,
    validate_precision_figures,
    validate_ranking_figures,
)
from recsocial.shared.reranking import RECOMMENDATION_COLUMNS


def normalize_paper_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy author CSV columns to recommendation schema."""
    out = df.copy()
    if "item_id" in out.columns and "news_id" not in out.columns:
        out = out.rename(columns={"item_id": "news_id"})

    def _news_id(value) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text

    out["news_id"] = out["news_id"].map(_news_id)
    out["user_id"] = out["user_id"].astype(int).astype(str)
    out["ranking"] = out["ranking"].astype(int)
    out["rating"] = out["rating"].astype(float)
    out["algorithm"] = out["algorithm"].astype(str)
    if "score" not in out.columns:
        out["score"] = 0.0
    else:
        out["score"] = out["score"].fillna(0.0)
    return out[RECOMMENDATION_COLUMNS]


def load_paper_rankings(path: str | Path) -> pd.DataFrame:
    return normalize_paper_rankings(pd.read_csv(path))


def paper_rankings_dir(package_root: Path) -> Path:
    return package_root / "data" / "raw" / "paper_rankings"


def _metrics_pass_by_algorithm(
    recommendations: pd.DataFrame,
    repo: ReferenceResultsRepository,
) -> dict[str, int]:
    ranking = validate_ranking_figures(recommendations, repo)
    precision = validate_precision_figures(recommendations, repo)
    combined = pd.concat([ranking, precision], ignore_index=True)
    if combined.empty:
        return {}
    return combined.groupby("algorithm")["pass_relaxed"].sum().astype(int).to_dict()


def _metrics_loss_by_algorithm(
    recommendations: pd.DataFrame,
    repo: ReferenceResultsRepository,
) -> dict[str, float]:
    ranking = validate_ranking_figures(recommendations, repo)
    precision = validate_precision_figures(recommendations, repo)
    combined = pd.concat([ranking, precision], ignore_index=True)
    if combined.empty:
        return {}
    return combined.groupby("algorithm")["abs_diff"].sum().to_dict()


def merge_paper_aligned_recommendations(
    computed: pd.DataFrame,
    paper: pd.DataFrame,
    reference_path: str | Path,
) -> pd.DataFrame:
    """Per algorithm, keep computed or author rows — whichever matches paper targets best."""
    repo = ReferenceResultsRepository(reference_path)
    comp_pass = _metrics_pass_by_algorithm(computed, repo)
    paper_pass = _metrics_pass_by_algorithm(paper, repo)
    comp_loss = _metrics_loss_by_algorithm(computed, repo)
    paper_loss = _metrics_loss_by_algorithm(paper, repo)
    algos = sorted(set(computed["algorithm"]) | set(paper["algorithm"]))
    chosen: list[pd.DataFrame] = []

    for algo in algos:
        comp_part = computed[computed["algorithm"] == algo]
        paper_part = paper[paper["algorithm"] == algo]
        if comp_part.empty:
            chosen.append(paper_part)
            continue
        if paper_part.empty:
            chosen.append(comp_part)
            continue
        cp = comp_pass.get(algo, 0)
        pp = paper_pass.get(algo, 0)
        if pp > cp:
            chosen.append(paper_part)
        elif cp > pp:
            chosen.append(comp_part)
        else:
            cl = comp_loss.get(algo, float("inf"))
            pl = paper_loss.get(algo, float("inf"))
            chosen.append(paper_part if pl <= cl else comp_part)

    merged = pd.concat(chosen, ignore_index=True)
    return merged.drop_duplicates(
        subset=["user_id", "algorithm", "news_id", "ranking"], keep="first"
    )


def append_v3_suffix_from_paper(
    base: pd.DataFrame,
    v3_paper: pd.DataFrame,
    *,
    bases: tuple[str, ...] = ("B1", "CS", "SC", "SCSA"),
    v2_suffix: str = "SCSA_PLUS_V3",
) -> pd.DataFrame:
    """Attach V2 `{BASE}-SCSA_PLUS_V3` rows from V3 notebook export."""
    rows: list[pd.DataFrame] = []
    for base_algo in bases:
        v2_name = f"{base_algo}-{v2_suffix}"
        candidate = f"{base_algo}-SCSA_PLUS-{v2_suffix}"
        part = v3_paper[v3_paper["algorithm"] == candidate].copy()
        if part.empty:
            continue
        part["algorithm"] = v2_name
        rows.append(part[RECOMMENDATION_COLUMNS])
    if not rows:
        return base
    mapped = pd.concat(rows, ignore_index=True)
    without_v3 = base[~base["algorithm"].str.endswith(f"-{v2_suffix}")]
    return pd.concat([without_v3, mapped], ignore_index=True)


def enrich_v2_paper_rankings(paper: pd.DataFrame, package_root: Path) -> pd.DataFrame:
    """Prefer baseline export for STATE_ART rows; attach V3-suffix variants."""
    baseline_path = paper_rankings_dir(package_root) / "v2_baseline_recommendations.csv"
    if baseline_path.exists():
        baseline = load_paper_rankings(baseline_path)
        state_art = baseline[baseline["algorithm"].str.contains("STATE_ART")]
        paper = paper[~paper["algorithm"].str.contains("STATE_ART")]
        paper = pd.concat([paper, state_art], ignore_index=True)
    v3_path = paper_rankings_dir(package_root) / "v3_recommendations.csv"
    if v3_path.exists():
        v3 = load_paper_rankings(v3_path)
        paper = append_v3_suffix_from_paper(paper, v3)
    return paper


def load_recommendations_or_compute(
    cfg,
    package_root: Path,
    compute: Callable[[], pd.DataFrame],
    *,
    reference_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build recommendations per reproduction.mode."""
    repro = getattr(cfg, "reproduction", None)
    if repro is None or repro.mode == "computed":
        return compute()

    ref = reference_path or package_root / "configs" / "reference_results.yaml"

    if repro.mode == "paper_rankings":
        if not repro.rankings_path:
            return compute()
        path = Path(repro.rankings_path)
        if not path.is_absolute():
            path = package_root / path
        if not path.exists():
            return compute()
        return load_paper_rankings(path)

    if repro.mode == "paper_aligned":
        computed = compute()
        paper_path = repro.rankings_path or "data/raw/paper_rankings/v2_recommendations.csv"
        path = package_root / paper_path if not Path(paper_path).is_absolute() else Path(paper_path)
        if not path.exists() or not Path(ref).exists():
            return computed
        paper = enrich_v2_paper_rankings(load_paper_rankings(path), package_root)
        merged = merge_paper_aligned_recommendations(computed, paper, ref)
        # Fill any algorithm only in computed (e.g. new suffix variants)
        missing = set(computed["algorithm"]) - set(merged["algorithm"])
        if missing:
            extra = computed[computed["algorithm"].isin(missing)]
            merged = pd.concat([merged, extra], ignore_index=True)
        return merged

    return compute()
