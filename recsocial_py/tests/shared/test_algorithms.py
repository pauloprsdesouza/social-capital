"""Tests for canonical algorithm naming."""

from recsocial.shared.algorithms import (
    BASE_ALGORITHMS,
    V1_PAPER_LABELS,
    paper_label_v1,
    parse_base_algorithm,
    rerank_variant,
)


def test_paper_label_v1() -> None:
    assert paper_label_v1("CS") == "CS-PLUS"
    assert paper_label_v1("SCSA") == "SC+SA"


def test_rerank_variant() -> None:
    assert rerank_variant("B1", "state_art") == "B1-STATE_ART"
    assert rerank_variant("SC", "scsa_plus") == "SC-SCSA_PLUS"


def test_parse_base_algorithm() -> None:
    assert parse_base_algorithm("B1-STATE_ART") == "B1"
    assert parse_base_algorithm("SC-SCSA_PLUS_V3") == "SC"


def test_base_algorithms_tuple() -> None:
    assert set(BASE_ALGORITHMS) == {"B1", "CS", "SC", "SCSA"}
    assert len(V1_PAPER_LABELS) == 4
