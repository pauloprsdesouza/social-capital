"""Unit tests for reference validation helpers."""

from __future__ import annotations

from recsocial.shared.reference_validation import resolve_impl_algorithm, resolve_winner_name


def test_resolve_impl_algorithm_b1_scsa_plus() -> None:
    assert resolve_impl_algorithm("figure_3_b1", "B1_SCSA_PLUS") == "B1-SCSA_PLUS"


def test_resolve_impl_algorithm_scsa_plus_v3() -> None:
    assert resolve_impl_algorithm("figure_4_cs", "SCSA_PLUS_V3") == "CS-SCSA_PLUS_V3"


def test_resolve_winner_name() -> None:
    assert resolve_winner_name("figure_8_cs", "SCSA_PLUS_V3") == "CS-SCSA_PLUS_V3"
    assert resolve_winner_name("figure_3_b1", "B1-SCSA_PLUS") == "B1-SCSA_PLUS"
