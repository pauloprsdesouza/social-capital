"""Canonical algorithm names and display labels used across all slices.

Use this module whenever mapping between:
- internal codes (B1, CS, SC, SCSA)
- paper labels (CS-PLUS, SC+SA)
- reranked variant names (B1-STATE_ART, SC-SCSA_PLUS_V3)
"""

from __future__ import annotations

BASE_ALGORITHMS: tuple[str, ...] = ("B1", "CS", "SC", "SCSA")

# Internal code → paper / report label (V1 FedCSIS)
V1_PAPER_LABELS: dict[str, str] = {
    "B1": "B1",
    "CS": "CS-PLUS",
    "SC": "SC",
    "SCSA": "SC+SA",
}

# Reverse lookup for validation against v1.yaml paper_targets
V1_FROM_PAPER_LABEL: dict[str, str] = {v: k for k, v in V1_PAPER_LABELS.items()}

# V3 paper §26 headline alias: chart says "SCSA-PLUS", implementation uses base SC
V3_HEADLINE_ALIASES: dict[str, str] = {
    "SCSA_PLUS": "SC",
}

RERANK_SUFFIXES: dict[str, str] = {
    "state_art": "STATE_ART",
    "scsa_plus": "SCSA_PLUS",
    "scsa_plus_v3": "SCSA_PLUS_V3",
}


# Display order for V1 figures and reports (FedCSIS paper)
V1_PAPER_ALGORITHM_ORDER: tuple[str, ...] = tuple(V1_PAPER_LABELS.values())


def paper_label_v1(code: str) -> str:
    """Map internal algorithm code to FedCSIS paper label."""
    return V1_PAPER_LABELS.get(code, code)


def rerank_variant(base: str, suffix_key: str) -> str:
    """Build a reranked algorithm name, e.g. rerank_variant('B1', 'state_art') → 'B1-STATE_ART'."""
    suffix = RERANK_SUFFIXES[suffix_key]
    return f"{base}-{suffix}"


def parse_base_algorithm(variant: str) -> str:
    """Extract base algorithm from a variant name (B1-STATE_ART → B1)."""
    for base in BASE_ALGORITHMS:
        if variant == base or variant.startswith(f"{base}-"):
            return base
    return variant.split("-", 1)[0]
