"""Assert reproduction metrics match paper reference values (all slices)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recsocial.shared.paper_validation import validate_all_papers


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def validation() -> dict:
    result = validate_all_papers(PACKAGE_ROOT)
    if all(s.n_checks == 0 for s in result["summaries"]):
        pytest.skip("No report CSVs found — run `python -m recsocial.cli run all` first")
    return result


def test_v1_trial_metrics_match_fedcsis_paper(validation: dict) -> None:
    v1 = validation["v1"]
    df = v1["details"]
    if df.empty:
        pytest.skip("Missing V1 trial_metrics_summary.csv")
    assert v1["summary"].n_pass_relaxed == v1["summary"].n_checks, (
        f"V1: {v1['summary'].n_pass_relaxed}/{v1['summary'].n_checks} within tolerance"
    )


def test_v3_headline_scsa_plus_matches_paper(validation: dict) -> None:
    v3 = validation["v3"]
    df = v3["details"]
    if df.empty:
        pytest.skip("Missing V3 v3_metrics_summary.csv")
    headline = df[df["paper_label"] == "SCSA_PLUS"]
    assert not headline.empty, "Missing SCSA_PLUS (SC) headline checks"
    assert headline["pass_relaxed"].all(), (
        f"SCSA_PLUS headline: {headline[~headline['pass_relaxed']].to_string()}"
    )


def test_v3_relaxed_reproduction_majority(validation: dict) -> None:
    v3 = validation["v3"]
    if v3["summary"].n_checks == 0:
        pytest.skip("Missing V3 data")
    rate = v3["summary"].n_pass_relaxed / v3["summary"].n_checks
    assert rate >= 0.8, f"V3: only {rate:.0%} within tolerance"


def test_v2_relaxed_reproduction_majority(validation: dict) -> None:
    v2 = validation["v2"]
    df = v2["details"]
    if df.empty:
        pytest.skip("Missing V2 recommendations")
    rate = v2["summary"].n_pass_relaxed / v2["summary"].n_checks
    assert rate >= 0.9, f"V2: only {rate:.0%} within relaxed tolerance"


def test_all_slices_have_validation_data(validation: dict) -> None:
    for s in validation["summaries"]:
        assert s.n_checks > 0, f"{s.slice_id} has no validation checks"
