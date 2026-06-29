"""Matplotlib styling for publication-style figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = {
    "B1": "#4C72B0",
    "CS": "#55A868",
    "CS-PLUS": "#55A868",
    "SC": "#C44E52",
    "SC+SA": "#8172B3",
    "SCSA": "#8172B3",
    "SCSA_PLUS": "#DD8452",
    "STATE_ART": "#937860",
    "SCSA_PLUS": "#DD8452",
    "default": "#4C72B0",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def color_for_algorithm(name: str) -> str:
    for key, color in PALETTE.items():
        if key in name or name.startswith(key):
            return color
    return PALETTE["default"]
