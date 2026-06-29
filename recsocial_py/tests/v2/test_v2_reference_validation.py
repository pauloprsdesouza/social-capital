"""Validate V2 AMCIS reproduction against reference_results.yaml."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsocial.shared.reference_validation import (
    ReferenceResultsRepository,
    run_v2_reference_validation,
    validate_anomaly_preservation,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PACKAGE_ROOT / "configs" / "reference_results.yaml"
RECOMMENDATIONS_PATH = PACKAGE_ROOT / "reports" / "v2" / "v2_recommendations.csv"


@pytest.fixture(scope="module")
def validation_result() -> dict:
    if not RECOMMENDATIONS_PATH.exists():
        pytest.skip("Run V2 experiment first (v2_recommendations.csv missing)")
    recs = pd.read_csv(RECOMMENDATIONS_PATH)
    return run_v2_reference_validation(
        recs,
        REFERENCE_PATH,
        PACKAGE_ROOT / "reports" / "v2",
    )


def test_reference_yaml_anomaly_preserved() -> None:
    repo = ReferenceResultsRepository(REFERENCE_PATH)
    assert validate_anomaly_preservation(repo)


def test_v2_ndcg_binary_matches_figure_3_b1_scsa_plus(validation_result: dict) -> None:
    ranking: pd.DataFrame = validation_result["ranking"]
    row = ranking[
        (ranking.figure == "figure_3_b1")
        & (ranking.algorithm == "B1-SCSA_PLUS")
        & (ranking.metric == "NDCG_10")
    ].iloc[0]
    assert row["pass_relaxed"], (
        f"B1-SCSA_PLUS NDCG@10: {row.actual:.3f} vs paper {row.expected:.3f}"
    )


def test_v2_state_art_map_matches_figure_3_relaxed(validation_result: dict) -> None:
    ranking: pd.DataFrame = validation_result["ranking"]
    row = ranking[
        (ranking.figure == "figure_3_b1")
        & (ranking.algorithm == "B1-STATE_ART")
        & (ranking.metric == "MAP_10")
    ].iloc[0]
    assert row["pass_relaxed"], (
        f"B1-STATE_ART MAP@10: {row.actual:.3f} vs paper {row.expected:.3f}"
    )


def test_v2_precision_p1_matches_figure_7_b1_scsa_plus_relaxed(
    validation_result: dict,
) -> None:
    precision: pd.DataFrame = validation_result["precision"]
    row = precision[
        (precision.figure == "figure_7_b1")
        & (precision.algorithm == "B1-SCSA_PLUS")
        & (precision.metric == "P_1")
    ].iloc[0]
    assert row["pass_relaxed"], (
        f"B1-SCSA_PLUS P@1: {row.actual:.3f} vs paper {row.expected:.3f}"
    )


def test_v2_figure_3_mrr_winner(validation_result: dict) -> None:
    winners: pd.DataFrame = validation_result["winners"]
    row = winners[
        (winners.figure == "figure_3_b1") & (winners.metric == "MRR")
    ].iloc[0]
    assert row["pass"], f"MRR winner expected {row.expected_winner}, got {row.actual_winner}"


def test_v2_figure_3_map_winner(validation_result: dict) -> None:
    winners: pd.DataFrame = validation_result["winners"]
    row = winners[
        (winners.figure == "figure_3_b1") & (winners.metric == "MAP_10")
    ].iloc[0]
    assert row["pass"]


def test_v2_figure_7_p1_winner(validation_result: dict) -> None:
    winners: pd.DataFrame = validation_result["winners"]
    row = winners[
        (winners.figure == "figure_7_b1") & (winners.metric == "P_1")
    ].iloc[0]
    assert row["pass"], f"P@1 winner expected {row.expected_winner}, got {row.actual_winner}"


def test_v2_relaxed_reproduction_majority(validation_result: dict) -> None:
    ranking: pd.DataFrame = validation_result["ranking"]
    precision: pd.DataFrame = validation_result["precision"]
    combined = pd.concat([ranking, precision], ignore_index=True)
    rate = combined["pass_relaxed"].mean()
    assert rate >= 0.5, f"Only {rate:.0%} of metric checks within relaxed tolerance"
